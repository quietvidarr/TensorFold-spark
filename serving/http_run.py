"""Upstream TensorFold HTTP run, owned GLM image-continuation seam only."""
def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
        prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``."""

    arrived = time.perf_counter()               # /metrics times a request from its arrival, queue wait included
    prepared = prepared if prepared is not None else self.prepare(body, chat)
    prompt, max_tokens = prepared.prompt, prepared.max_tokens
    tools, thinking = prepared.tools, prepared.thinking
    policy = ToolCallPolicy(body)
    sampling = prepared.sampling
    # end tokens end the reply and stay out of its text, unless it asks ignore_eos of an engine that reads it
    takes_stop_eos = "stop_eos" in inspect.signature(self.engine.generate).parameters
    ends = () if prepared.ignore_eos and (self.reads_ignore_eos or takes_stop_eos) else tuple(self.engine.eos)
    stops = StopStrings(prepared.stop, self.tok, ends)
    out: list[int] = []
    sent = {"reasoning": 0, "content": 0}
    stopped = {"client": False, "stop": False}
    failed: list[Exception] = []
    stream = StreamDecoder(self.tok, ends)
    serving: list[Any] = [None]                 # the running request, so /metrics sees its first token

    def visible(finished: bool) -> tuple[str, str]:
        raw = stream.final() if finished else stream.text
        # stop strings match the generated text, reasoning included, before it is split (as on the Mac)
        raw = stops.visible(raw, partial=not finished) if stops.strings else raw
        if chat and thinking:
            reasoning, answer = split_thinking(raw, finished=finished)
        else:
            reasoning, answer = "", raw
        if tools:
            answer = (policy.content(answer, finished=finished) if policy.single
                      else hide_tool_calls(answer, finished=finished))
        return reasoning, answer

    def on_tokens(new: list[int]) -> bool:
        # True stops the engine after this round; engines that finish on both ranks keep calling and get True
        if stopped["client"] or stopped["stop"] or failed:
            return True
        try:
            if stops.strings:
                kept = []
                for token in new:             # token by token: the round's width cannot move the cut
                    kept.append(token)
                    out.append(token)
                    if stops.hit(out):
                        stopped["stop"] = True
                        break
                new = kept
            else:
                out.extend(new)
            stream.add(new)
            reasoning, answer = visible(False)
            delta: dict[str, Any] = {}
            if len(reasoning) > sent["reasoning"]:
                delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                sent["reasoning"] = len(reasoning)
            if len(answer) > sent["content"]:
                delta["content"] = answer[sent["content"]:]
                sent["content"] = len(answer)
            if delta and not emit(delta):
                stopped["client"] = True
            elif cancelled is not None and cancelled():     # every round, with or without new text
                stopped["client"] = True
            if serving[0] is not None:
                serving[0].saw()
        except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
            failed.append(exc)
            return True
        return stopped["client"] or stopped["stop"]

    draft = body.get("draft", True) is not False
    gate = self._call_gate(prompt, tools) if tools and tool_choice_requires_call(body.get("tool_choice")) else None

    options: dict[str, Any] = {} if draft else {"draft": False}
    if takes_stop_eos:
        options["stop_eos"] = not prepared.ignore_eos
    shaped = prepared.grammar is not None or prepared.think_budget > 0
    think_end = self.tok.token_to_id("</think>") if chat and thinking and shaped else None
    budget = self._think_budget(prepared, think_end)
    # priority "background" (or a session-title request): after the others, as on the Mac
    background = body.get("priority") == "background" or (chat and is_title_request(body.get("messages"), tools))
    concurrent = getattr(self.engine, "concurrent", False)
    turns = None if concurrent else self._turns()
    if concurrent and background and "background" in inspect.signature(self.engine.generate).parameters:
        options["background"] = True            # the engine's scheduler orders its lanes and prompts
    # one engine at a time: a background reply yields between rounds (not on two ranks, which decode to the end)
    # A yield restart would reset the matcher in the middle of a JSON object.
    # Structured requests retain background queue priority, but run atomically.
    yielding = prepared.grammar is None and background and turns is not None and getattr(self.engine, "tp", 1) == 1
    gates = [g for g in (gate, budget, Yield(turns) if yielding else None) if g is not None]

    cached: list[int] = []              # the prompt tokens the first run found cached (usage's cached_tokens)

    def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
        if yielding and cached:                 # a run after a cut: foreground requests waiting go first
            turns.give()
            turns.take(True, cancelled)
        extra = dict(options)
        extra["landing_state"] = prepared.landing_state
        extra["prefix_cache"] = prepared.prefix_cache     # resume from the live prompt state, unless the request opts out
        extra["cancelled"] = cancelled                    # a prompt whose client has left stops between prefill chunks
        if prepared.grammar is not None:    # response_format: a fresh grammar state, after </think> when thinking
            spec, compiled = prepared.grammar
            # a run after the thinking budget's close (it ends at </think> under a grammar) starts in the grammar
            end = None if think_end is None or think_end in ids[len(prompt):] else think_end
            extra["constraint"] = self._grammars().constraint(compiled, think_end=end, spec=spec)
        if prepared.vision is not None:          # a gate's continuation keeps the images, positions extended
            from tensorfold.vision.qwen_processing import continued

            same = list(ids) == list(prepared.vision.token_ids)
            extra["vision"] = prepared.vision  # GLM image spans stay fixed for text-only continuation
        stats = self.engine.generate(ids, count, sampling, feed, **extra)
        if not cached:
            cached.append(int((stats or {}).get("cached") or 0))
        return stats

    # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
    if turns is not None:
        turns.take(background, cancelled)
    try:
        if cancelled is not None and cancelled():                # the client left while this request waited
            raise RequestCancelled("the client left before the request started")
        with health.of(self).running(len(prompt), out, arrived) as request:  # /health reads ``out``; rounds never call in
            serving[0] = request
            try:
                stats = request.stats = generate_gated(generate, prompt, max_tokens, gates, on_tokens)
            finally:
                serving[0] = None
    finally:
        if turns is not None:
            turns.give()
    if failed:
        raise failed[0]
    if stopped["client"]:                                        # as the Mac server: nothing more is written
        raise RequestCancelled("the client left during the reply")
    stats = {**(stats or {}), "token_sha": token_sha(out)}
    reasoning, answer = visible(True)
    final: dict[str, Any] = {}
    if len(reasoning) > sent["reasoning"]:
        final["reasoning_content"] = reasoning[sent["reasoning"]:]
    text = stops.visible(self.tok.decode([t for t in out if t not in ends], skip_special_tokens=False))
    raw_answer = split_thinking(text, finished=True)[1] if chat and thinking else text
    content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
    content = policy.content(content) if tools else content
    tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
    if tail:
        final["content"] = tail
    finish = "tool_calls" if calls else ("stop" if stopped["stop"] or (out and out[-1] in ends) else "length")
    termination = (stats or {}).get("thinking_loop", {}).get("termination_reason")
    if termination == "answer_cap":
        finish = "length"
    elif termination in ("answer_repeat_ngram", "answer_low_variety", "answer_collapse"):
        finish = "stop"  # explicit protective stop; metadata distinguishes from EOS completion
    if prepared.grammar is not None:
        # Caps/guards may return an incomplete JSON prefix, never a claimed
        # successful structured stop. No output stripping, repair or prompt trick.
        finish = "stop" if stats.get("structured_output", {}).get("grammar_terminated") else "length"
    if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
        stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
    return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
            "prompt_tokens": len(prompt), "completion_tokens": len(out), "cached_tokens": (cached or [0])[0],
            "reasoning_tokens": reasoning_count(out, self.tok.token_to_id("</think>") if chat and thinking else None),
            "stats": stats}

