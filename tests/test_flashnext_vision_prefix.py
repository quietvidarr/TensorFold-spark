"""CPU identity, admission isolation and rotary restoration for image-prefix reuse."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import pytest
from tensorfold.vision.qwen_processing import PreparedVisionPrompt
from tensorfold.families.qwen4_exp.cuda import prefixes, vision_prefix, decode
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.multi_fill import PromptPasses


def prepared(extra=0):
    n = 12 + extra
    return PreparedVisionPrompt(tuple(range(n)), np.arange(16,dtype=np.float32).reshape(4,4),
        np.array([[1,2,2]],dtype=np.int64), np.broadcast_to(np.arange(n,dtype=np.int32),(3,1,n)).copy(),
        -2, ((3,7),), ('image-a',))


def test_text_extension_preserves_media_namespace():
    assert vision_prefix.cache_key(prepared()) == vision_prefix.cache_key(prepared(8))


def test_raw_content_pixel_grid_span_position_and_delta_changes_all_miss():
    p = prepared(); key = vision_prefix.cache_key(p)
    variants = [replace(p,image_hashes=('image-b',)), replace(p,pixel_values=p.pixel_values+1),
                replace(p,image_grid_thw=np.array([[1,4,1]],dtype=np.int64)),
                replace(p,image_spans=((4,8),)), replace(p,position_ids=p.position_ids+1),
                replace(p,rope_delta=-1),replace(p,pixel_values=p.pixel_values.astype(np.float64))]
    assert all(vision_prefix.cache_key(q) != key for q in variants)


def test_added_media_has_distinct_namespace():
    p=prepared(); q=replace(p,image_hashes=('image-a','image-b'),image_spans=((3,7),(8,10)),
                          image_grid_thw=np.array([[1,2,2],[1,2,2]]),pixel_values=np.concatenate([p.pixel_values,p.pixel_values]))
    assert vision_prefix.cache_key(p)!=vision_prefix.cache_key(q)


def test_video_keeps_cold_path_and_missing_identity_not_reused():
    p=prepared(); video=replace(p,video_hashes=('clip',))
    assert vision_prefix.cache_key(video) is None
    assert not vision_prefix.reusable(video,None)
    assert vision_prefix.reusable(None,None)
    missing=replace(p,image_hashes=())
    assert vision_prefix.cache_key(missing) is None
    unknown=SimpleNamespace()
    assert vision_prefix.cache_key(unknown) is None
    assert not vision_prefix.reusable(unknown,None)


def test_prefix_selector_never_crosses_media_or_text_namespaces():
    ids=[1,2,3]; st=object()
    entries=[(ids,st,{'vision_cache_key':'a'},None)]
    assert prefixes._best(entries,ids+[4],cache_key='a') is entries[0]
    assert prefixes._best(entries,ids+[4],cache_key='b') is None
    assert prefixes._best(entries,ids+[4]) is None
    text=[(ids,st,{},None)]
    assert prefixes._best(text,ids+[4]) is text[0]
    assert prefixes._best(text,ids+[4],cache_key='a') is None


def test_same_token_ids_different_images_coexist_without_displacement():
    a,b=object(),object(); ids=[1,2]
    owner=SimpleNamespace(kept=[(ids,a,{'vision_cache_key':'a'},None)],keep=8,free=[],_busy=lambda:set())
    prefixes.remember(owner,ids,b,{'vision_cache_key':'b'},None)
    assert len(owner.kept)==2 and owner.free==[]


def state():
    st=object.__new__(State)
    st.capacity=256;st.kv_dtype='bf16'
    st.rec=torch.zeros((2,1,1,1,1));st.conv=torch.zeros((1,1,1));st.cur=[0]
    st.ple_tail=torch.zeros((1,1));st.ple_history=None;st.ple_last=None
    st.pos_dev=torch.zeros(1,dtype=torch.int32);st.rope_delta_dev=torch.zeros(1,dtype=torch.int32)
    st.mtp_pos=torch.zeros(1,dtype=torch.int32);st.mtp_len=3;st.mtp_drafted=0
    st.pos=4;st.rope_delta=-2;st.image_positions=torch.arange(36,dtype=torch.int32).view(12,3)
    st.vision_cache_key='namespace';st.image_rows=();st.image_features=None
    return st


def test_snapshot_restores_visual_rotary_and_identity_without_features():
    st=state(); snap=st.snapshot(); target=state()
    target.rope_delta=0;target.image_positions=None;target.vision_cache_key=None
    target.restore(snap)
    assert target.rope_delta==-2 and target.rope_delta_dev.item()==-2
    assert target.image_positions is snap['image_positions']
    assert target.vision_cache_key=='namespace' and target.image_features is None


def test_restore_legacy_text_snapshot_clears_previous_visual_metadata():
    st=state(); snap=st.snapshot()
    for k in ['image_positions','rope_delta','vision_cache_key']:snap.pop(k)
    st.restore(snap)
    assert st.image_positions is None and st.rope_delta==0 and st.vision_cache_key is None


def test_mtp_tail_absorb_sees_restored_image_rotary_before_reattachment():
    st=state();snap=st.snapshot();st.image_positions=None;st.rope_delta=0
    e=SimpleNamespace(st=st,w=SimpleNamespace(mtp=True),mbuf=object(),pbuf=object())
    def absorb(w,got,pb,tokens,tail):
        assert got.image_positions is snap['image_positions'] and got.rope_delta==-2
    with patch.object(decode,'mtp_forward',side_effect=absorb) as call:
        assert decode.prefill_begin(e,list(range(12)),resume={'state':snap,'tail':'tail'})==4
        call.assert_called_once()


def test_only_keyed_visual_states_are_keepable():
    st=state();s=SimpleNamespace(st=st,draft=True,prompt=list(range(12)))
    assert PromptPasses._keep_at(s)==11
    st.vision_cache_key=None
    assert PromptPasses._keep_at(s) is None


def test_namespace_includes_readonly_arrays_without_mutation():
    p=prepared(); arrays=[p.pixel_values,p.image_grid_thw,p.position_ids]
    for a in arrays:a.setflags(write=False)
    before=[a.copy() for a in arrays]
    assert vision_prefix.cache_key(p)
    assert all(np.array_equal(a,b) for a,b in zip(arrays,before))


def test_mismatched_prompt_or_geometry_fails_before_cache_selection():
    p=prepared()
    for q,prompt in [(p,[999]),(replace(p,position_ids=p.position_ids[:,:,:5]),p.token_ids),
                     (replace(p,pixel_values=p.pixel_values[:2]),p.token_ids)]:
        with pytest.raises(ValueError):vision_prefix.cache_key(q,prompt)


@pytest.mark.parametrize('destination_visual', [None, torch.zeros((2,3),dtype=torch.int32)])
def test_dynamic_visual_relocation_preserves_fixed_buffer_addresses(destination_visual):
    source,target=state(),state();target.image_positions=destination_visual
    pointers={k:getattr(target,k).data_ptr() for k in ['rec','conv','pos_dev','rope_delta_dev','mtp_pos']}
    target.copy_from(source)
    assert target.image_positions is source.image_positions
    assert target.vision_cache_key==source.vision_cache_key and target.rope_delta==-2
    assert all(getattr(target,k).data_ptr()==v for k,v in pointers.items())
