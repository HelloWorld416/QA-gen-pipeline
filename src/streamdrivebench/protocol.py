from .common import obj_schema, STRING, INTEGER, BOOL, INTS
from . import checks as P
import re

CANDIDATE = obj_schema(dict(family={'type':'string','enum':['HGI','AD','ETP']},
    query_second=INTEGER,history_start_second=INTEGER,horizon_seconds=INTEGER,question=P.QUESTION))
EVENT = obj_schema(dict(actor=STRING,behavior=STRING,before_state=STRING,after_state=STRING,
    approximate_transition=INTEGER,evidence_frames=INTS))
PROPOSAL = obj_schema(dict(events=P.array(EVENT),coverage_notes=STRING))
DISCOVER_PROMPT = """Discover driving events ONLY; do not generate questions or answers. Images cover exactly one ownership window [start,end_exclusive), without neighboring context. Select at most max_events distinct salient events with a visible atomic transition and before/after evidence INSIDE the window. Prefer interactions, stop/start, crossing, turning and signal changes. Use absolute integer seconds. No hidden intent, metric speed or inferred telemetry. Do not duplicate the same actor's same atomic transition or relabel an ongoing state. Rank by clarity and return no events if unsupported. Candidates require independent boundary confirmation."""
OMISSION = obj_schema(dict(family={'type':'string','enum':['HGI','AD','ETP']},reason=STRING))
AUTHOR = obj_schema(dict(question_sets=P.array(obj_schema(dict(
    event_id=STRING,questions=P.array(CANDIDATE),omissions=P.array(OMISSION))))))
AUTHOR_PROMPT = """Generate driving QA ONLY for the supplied confirmed_events. Their boundaries have already passed independent verification. Do not invent events, change their identifiers or boundaries, or attach questions to a rejected event. All events share the supplied window images: return exactly one question_set per confirmed event_id.
For EACH event actively attempt one question for EACH requested_families entry (default HGI, AD, ETP) concerning the same identifiable actor and continuous behavior episode. Never silently omit a family: provide either exactly one question or an explicit evidence-based omission reason for each requested family. Never generate a family absent from requested_families. Do not invent unsupported questions to fill quotas. Do not output answers. Avoid duplicate model inputs across events.
Every question specifies history_start_second: choose only the history needed for this question, from ownership_start through query_second, with query_second-history_start_second <= 30. A full past 30 seconds is NOT required. HGI needs at least one earlier observation; AD/ETP may use zero history if sufficient. HGI follows this selected historical evidence within this window and concerns current state/identity genuinely requiring history. AD predicts observable behavior exactly 3 seconds later; the query precedes the confirmed transition and both query and forecast lie inside this window. ETP injection precedes the confirmed transition inside this window and requires a full 30-second monitor within the video; it can extend beyond the discovery window. Use video_num_seconds to omit ETP when query_second+30 >= video_num_seconds. Do not assume unseen future outcomes.
ETP must monitor ONE atomic driving-state condition. Before it occurs, and throughout the interval if it never occurs, remain SILENT. Never request a timeout/no-event answer. Loss of visibility is not a second trigger. Non-SILENT options describe possible answers when the single condition triggers, NOT timeout, non-occurrence or lost visibility. Do not instruct emitting a hardcoded answer text or letter: refer to selecting the corresponding option; final option letters are assigned later. Exactly one SILENT option. Prefer an observable transition of the confirmed event when supported.
Questions use current-time identifiable target descriptions, no future attributes, event labels or revealing wording. A conditional when/if does not assert future occurrence. HGI/AD have 2-4 alternatives and no SILENT; ETP has 2-4 non-SILENT alternatives plus SILENT. horizon_seconds=3 for AD, 0 otherwise. rationale_from_prefix cites only evidence available at query_second. Author sees the entire discovery window and confirmed event: this is offline outcome-aware authorship, not a causal predictor. Subsequent independent audits verify answers and causal validity."""


def windows(n):
    for start in range(0,n,30):
        yield start,min(start+30,n)


def prefix(start,t):
    return sorted(set(range(max(start,t-30),t+1,3)) | set(range(max(start,t-5),t+1)))


def preflight(candidate,start,end,n):
    family=candidate['family'];t=candidate['query_second'];h=candidate['horizon_seconds']
    errors=P.question_errors(candidate['question'],family)
    history=candidate.get('history_start_second',start)
    if type(history) is not int or not max(start,t-30)<=history<=t:errors.append('invalid_history_window')
    if family=='HGI' and history>=t:errors.append('no_history')
    if family!='ETP' and len(candidate['question']['choices'])>4:errors.append('too_many_choices')
    if not start<=t<end:errors.append('query_outside_window')
    if h!=(3 if family=='AD' else 0):errors.append('invalid_horizon')
    if family=='AD' and t+h>=end:errors.append('forecast_outside_window')
    if family=='HGI' and t<=start:errors.append('no_history')
    if family=='ETP' and t+30>=n:errors.append('incomplete_monitor')
    if family=='ETP':
        # High-precision guards for observed authoring failures, not a substitute
        # for the independent semantic injection check.
        q=candidate['question']
        if re.search(r'\b(?i:emit|output|return)\s+[A-Z]+_[A-Z_]+\b',q['question']):
            errors.append('hardcoded_answer_emission')
        if any(re.search(r'\bNO_[A-Z_]+_WITHIN_\d+_SECONDS\b',v) for v in q['choices']):
            errors.append('timeout_answer_option')
    return errors

COMMIT_PROMPT = "You are an online event monitor. You see only the video prefix available NOW, never later images, and are not told any ground-truth trigger or expected response. A question was injected at injection_second. Determine whether the provided evidence ALREADY warrants emitting its non-SILENT answer. Otherwise answer SILENT. Do not assume that the final image was selected because an event occurred. An uncertain or still evolving boundary is not permission to answer. may_answer_now=true only if the specified driving-state transition has observably occurred since injection and is sufficiently established from these images alone. Mere hints of a future maneuver do not suffice. If it already occurred earlier in this visible prefix, may_answer_now=true (this check does not ask you to repeat a previous response). evidence_decidable distinguishes a clear not-yet case from unresolvable visibility/identity. target_identifiable must hold throughout the relevant evidence. A turn means a maneuver changing the travel direction onto another road/access or a turnaround, not a lane change or ordinary road-following curvature, unless the question explicitly specifies otherwise. A complete stop needs temporal evidence from static roadside structure; do not infer it from one image or another vehicle's brake lights. If answering, choose an EXACT non-SILENT choice. Cite supplied evidence_frames, including the latest supplied second. Do not backdate an inferred onset using evidence that was unavailable at that earlier time."
COMMIT = obj_schema(dict(target_identifiable=BOOL, evidence_decidable=BOOL, may_answer_now=BOOL, answer=STRING, evidence_frames=INTS, reason=STRING))
