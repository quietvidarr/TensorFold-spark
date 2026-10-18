"""Owned GLM5.3 vision bridge; no TensorFold or checkpoint source modifications.

Importing this module imports neither torch nor ExLlama. Construct Frontend only
AFTER the TensorFold loader. HTTP prepare is CPU-only. Enter context ONLY inside
App's Turns lock, and run fresh, eager, MTP-off, serial generation inside it.
"""
from __future__ import annotations

import dataclasses
import os
import secrets
import sys
from contextlib import contextmanager
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class ImageSpan:
    """Absolute prompt rows containing features, excluding image start/end IDs."""
    start: int
    count: int
    grid_thw: tuple[int, int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class VisionRequest:
    frontend: Any
    token_ids: tuple[int, ...]
    images: tuple[Any, ...]              # decoded, oriented RGB PIL images on CPU
    spans: tuple[ImageSpan, ...]

    def continued(self, token_ids):
        """Owned GLM continuation: no Qwen MRoPE or virtual IDs are involved."""
        ids = tuple(token_ids)
        if ids[:len(self.token_ids)] != self.token_ids:
            raise ValueError('vision continuation must retain the entire prompt prefix')
        return dataclasses.replace(self, token_ids=ids)


class Frontend:
    max_images = 2
    max_image_tokens = 2048
    allow_urls = True

    def __init__(self, model_path, runtime):
        self.runtime = runtime
        self.engine, _, self.torch, _, _, _, self.forward, hooks, _ = runtime
        self.guard = hooks.guard
        self._active = False
        if 'torch' not in sys.modules:
            raise RuntimeError('construct vision Frontend after the TensorFold Torch loader')
        self.guard('before_native_vision_import')
        # -S runtimes need this explicit directory, NOT site.main()/sitecustomize.
        site_path = '/opt/sglang/lib/python3.12/site-packages'
        if site_path not in sys.path:
            sys.path.append(site_path)
        # ExLlama's import-time allocator preference must not change TensorFold's.
        previous = os.environ.get('EXL3_EXPANDABLE_SEGMENTS')
        os.environ['EXL3_EXPANDABLE_SEGMENTS'] = '0'
        try:
            from exllamav3.architecture.glm5_next import Glm5NextConfig, Glm5NextVisionModel
            from exllamav3.architecture.mm_processing.glm5_next import glm5_vision_canvas
            from exllamav3.tokenizer import Tokenizer
        finally:
            if previous is None:
                os.environ.pop('EXL3_EXPANDABLE_SEGMENTS', None)
            else:
                os.environ['EXL3_EXPANDABLE_SEGMENTS'] = previous
        from tensorfold.vision.images import ImageLimits
        self.limits = ImageLimits(max_images=self.max_images, max_pixels=4 * 1024 * 1024,
                                  max_total_pixels=8 * 1024 * 1024)
        self.canvas = glm5_vision_canvas
        self.guard('before_native_vision_config')
        # Config scans headers; only the vision Model below loads any tensors.
        self.config = Glm5NextConfig(str(model_path))
        self.config.vision_pp.max_image_tokens = self.max_image_tokens
        self.config.infer_params.vision_pinned = False
        v, pp = self.config.vision, self.config.vision_pp
        c = self.engine.w.cfg
        if (self.config.hidden_size != c.hidden or v.out_hidden_size != c.hidden
                or c.hidden != 4096 or self.config.vocab_size != c.vocab
                or pp.merge_size != v.spatial_merge_size or pp.patch_size != v.patch_size
                or pp.temporal_patch_size != v.temporal_patch_size
                or not 0 < pp.min_image_tokens <= self.max_image_tokens):
            raise ValueError('native vision/text geometry does not match the pinned GLM runtime')
        self.start_id = self.config.vision_start_token_id
        self.end_id = self.config.vision_end_token_id
        if not all(type(t) is int and 0 <= t < c.vocab for t in (self.start_id, self.end_id, 0, 1)):
            raise ValueError('image boundary/BOS/placeholder IDs must be ordinary vocabulary IDs')
        self.native_tokenizer = Tokenizer(self.config)
        self.model = Glm5NextVisionModel(self.config)
        # Direct construction bypasses Model.from_config's forward-map setup.
        self.model.prepare_layer_map()
        if not self.model.fwd_modules:
            raise RuntimeError('native vision forward map is empty')
        self.model.component = 'vision'
        self.guard('before_native_vision_load')
        try:
            with self.torch.inference_mode():
                self.model.load(device=self.engine.w.device, progressbar=False,
                                max_chunk_size=self.max_image_tokens * v.spatial_merge_size ** 2)
            self.torch.cuda.synchronize()
            self.guard('after_native_vision_load')
        except BaseException:
            self.model.unload()
            raise
        finally:
            self.config.stc.close()

    def _grid(self, image):
        """Predict native preprocessing's exact merged-token count without encoding."""
        pp, v = self.config.vision_pp, self.config.vision
        width, height = image.size
        factor = pp.patch_size * pp.merge_size
        h, w = self.canvas(pp.temporal_patch_size, height, width, pp.temporal_patch_size,
                           factor, pp.min_image_tokens, pp.max_image_tokens)
        grid = (1, h // pp.patch_size, w // pp.patch_size)
        count = grid[0] * grid[1] * grid[2] // (v.spatial_merge_size ** 2)
        if not 0 < count <= self.max_image_tokens:
            raise ValueError('native image canvas exceeds the serving image-token cap')
        return grid, count

    def prepare(self, body, app):
        """Return upstream PreparedRequest with safe IDs, BOS1 and absolute spans.

        Reuse App._prepare for the exact checkpoint template, tools, thinking,
        sampling and stopping options, after replacing media parts in a copy.
        No GPU work, native preprocessing tensors or feature vectors occur here.
        """
        from tensorfold.cuda.server import App
        from tensorfold.server.errors import RequestError
        from tensorfold.server.messages import validate_modalities
        from tensorfold.server.prompts import image_slot
        from tensorfold.vision.images import ImageInputError, load_images, split_images
        validate_modalities(body)
        try:
            messages, sources = split_images(body.get('messages'), limits=self.limits, allow_urls=True)
            if not sources:
                raise RequestError('vision preparation requires at least one image_url part')
            markers = []
            converted = []
            for message in messages:
                content = message.get('content')
                if isinstance(content, list):
                    text = []
                    for part in content:
                        if part['type'] == 'image':
                            marker = '<|tf_image_' + secrets.token_hex(24) + '|>'
                            markers.append(marker)
                            text.append(marker)
                        else:
                            text.append(part['text'])
                    converted.append({**message, 'content': ''.join(text)})
                else:
                    converted.append(message)
            # Tokenize each text/image boundary independently: GLM's BPE can merge
            # the '<' of a marker into preceding text/whitespace. A per-call proxy
            # keeps exact marker ID runs without mutating app.tok across HTTP threads.
            import re
            from types import SimpleNamespace
            pattern = '(' + '|'.join(re.escape(marker) for marker in markers) + ')'

            class MarkerTokenizer:
                def encode(self, text, *, add_special_tokens=False):
                    if add_special_tokens:
                        raise RequestError('vision marker encoding must not insert automatic BOS')
                    if any(text.count(marker) != 1 for marker in markers):
                        raise RequestError('chat template must preserve each image marker exactly once')
                    encoded = []
                    for piece in re.split(pattern, text):
                        if piece:
                            encoded.extend(app.tok.encode(piece, add_special_tokens=False).ids)
                    return SimpleNamespace(ids=encoded)

            class PrepareApp:
                tok = MarkerTokenizer()

                def __getattr__(self, name):
                    return getattr(app, name)

            # Deliberately call the base method, not ServingApp._prepare recursively.
            prepared = App._prepare(PrepareApp(), {**body, 'messages': converted}, True)
            tokens = list(prepared.prompt)
            if not tokens or tokens[0] != 1:
                tokens.insert(0, 1)
            slot = image_slot()
            try:
                decoded = load_images(sources, limits=self.limits, allow_urls=True)
                images = tuple(image.to_pil() for image in decoded)
            finally:
                slot.release()
            spans = []
            cursor = 0
            for marker, image in zip(markers, images, strict=True):
                needle = app.tok.encode(marker, add_special_tokens=False).ids
                if not needle:
                    raise RequestError('image marker tokenization is empty')
                matches = [i for i in range(len(tokens) - len(needle) + 1)
                           if tokens[i:i + len(needle)] == needle]
                if len(matches) != 1 or matches[0] < cursor:
                    raise RequestError('image marker must have exactly one ordered token occurrence')
                pos = matches[0]
                grid, count = self._grid(image)
                # Never pass MMEmbedding's -1 or billion-range virtual IDs to CUDA.
                tokens[pos:pos + len(needle)] = [self.start_id, *([0] * count), self.end_id]
                spans.append(ImageSpan(pos + 1, count, grid))
                cursor = pos + count + 2
            if any(type(t) is not int or not 0 <= t < self.engine.w.cfg.vocab for t in tokens):
                raise RequestError('vision prompt contains an out-of-vocabulary token ID')
            limit = app._context_limit()
            if len(tokens) > 84865 or (limit is not None and len(tokens) >= limit):
                raise RequestError('image-expanded prompt exceeds the serving input ceiling')
            request = VisionRequest(self, tuple(tokens), images, tuple(spans))
            return dataclasses.replace(prepared, prompt=tokens, vision=request,
                                       sampling=app.sampling_for(body, tokens))
        except (ImageInputError, ValueError) as exc:
            raise RequestError(str(exc)) from exc

    def encode(self, request):
        """Encode native quantized vision+merger, only inside the held context."""
        if not self._active or request.frontend is not self:
            raise RuntimeError('vision encoding must occur inside Frontend.context under Turns lock')
        features = []
        torch = self.torch
        with torch.inference_mode():
            for index, (image, span) in enumerate(zip(request.images, request.spans, strict=True)):
                self.guard(f'before_native_image_encode_{index}')
                mme = self.model.get_image_embeddings(self.native_tokenizer, image)
                tensor = mme.embeddings
                if (tuple(tensor.shape) != (span.count, self.engine.w.cfg.hidden)
                        or tuple(mme.grid_thw) != span.grid_thw
                        or not bool(torch.isfinite(tensor).all().item())):
                    raise ValueError(f'native vision mismatch: shape={tuple(tensor.shape)} expected={(span.count,self.engine.w.cfg.hidden)} grid={tuple(mme.grid_thw)} expected_grid={span.grid_thw} finite={bool(torch.isfinite(tensor).all().item())} dtype={tensor.dtype}')
                # Native returns embeddings.cpu(); preserve precision until copying to x.
                features.append(tensor.to(device=self.engine.w.device, dtype=self.engine.buf.x.dtype))
                del mme, tensor
                torch.cuda.synchronize()
                self.guard(f'after_native_image_encode_{index}')
        return features

    @contextmanager
    def context(self, request, prompt):
        """Temporarily inject features into only main buf/pbuf embedding rows.

        The caller owns Turns serialization, reset/fresh prefill, MTP-off and
        serial decoding. Positions use st.pos BEFORE commit. Restore on every
        exception, cancellation or successful completion. Prompt may extend a
        prefix for an owned GLM tool/budget continuation.
        """
        if request is None:
            yield
            return
        if request.frontend is not self or self._active:
            raise RuntimeError('wrong vision frontend or overlapping image generation')
        ids = tuple(prompt)
        if ids[:len(request.token_ids)] != request.token_ids:
            raise ValueError('vision prompt/BOS positions differ from prepared request')
        if (not ids or ids[0] != 1
                or any(type(t) is not int or not 0 <= t < self.engine.w.cfg.vocab for t in ids)):
            raise ValueError('vision generation requires BOS1 and vocabulary-safe IDs')
        for span in request.spans:
            if (span.start < 1 or span.count < 1 or span.start + span.count >= len(ids)
                    or ids[span.start - 1] != self.start_id
                    or ids[span.start:span.start + span.count] != (0,) * span.count
                    or ids[span.start + span.count] != self.end_id):
                raise ValueError('vision image span/boundary mismatch')
        e = self.engine
        glue = self.forward.glue
        original_embed, original_graphs = glue.embed, e.graphs
        main = [(b.x.data_ptr(), b.ids.data_ptr()) for b in (e.buf, e.pbuf) if b is not None]
        features = []
        seen = [set() for _ in request.spans]
        self._active = True
        try:
            self.guard('before_vision_request')
            e.graphs = None
            features = self.encode(request)

            def embed(token_ids, table, dims, copies, out):
                result = original_embed(token_ids, table, dims, copies, out)
                if (out.data_ptr(), token_ids.data_ptr()) not in main:
                    return result
                if table is not e.w.embed or dims != e.w.cfg.hidden or copies != e.w.cfg.streams:
                    raise RuntimeError('main embedding geometry changed during vision request')
                pos, rows = int(e.st.pos), int(token_ids.shape[0])
                with self.torch.inference_mode():
                    for index, (span, feature) in enumerate(zip(request.spans, features, strict=True)):
                        lo, hi = max(pos, span.start), min(pos + rows, span.start + span.count)
                        if hi > lo:
                            src = feature[lo - span.start:hi - span.start]
                            dst = out[lo - pos:hi - pos].view(hi - lo, copies, dims)
                            dst.copy_(src.unsqueeze(1).expand(-1, copies, -1))
                            seen[index].update(range(lo - span.start, hi - span.start))
                return result

            glue.embed = embed
            yield
            if any(len(rows) != span.count for rows, span in zip(seen, request.spans, strict=True)):
                raise RuntimeError('prefill did not inject every image feature row')
        finally:
            glue.embed = original_embed
            e.graphs = original_graphs
            features.clear()
            self._active = False
            self.torch.cuda.synchronize()
            self.guard('after_vision_request_restore')

    def close(self):
        if self._active:
            raise RuntimeError('cannot unload vision during generation')
        self.model.unload()
        self.config.stc.close()
