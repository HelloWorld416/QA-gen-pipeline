from streamdrivebench.protocol import preflight

def test_known_timeout_and_hardcoded_emission_rejected():
    candidate=dict(family='ETP',query_second=5,horizon_seconds=0,question=dict(question='Emit RIGHT_TURN_BEGINS; otherwise wait.',
        choices=['RIGHT_TURN_BEGINS','NO_RIGHT_TURN_WITHIN_30_SECONDS','SILENT'],target_description='car',predicate='turn',answer_slot='direction',rationale_from_prefix='car visible'))
    errors=preflight(candidate,0,30,60)
    assert 'hardcoded_answer_emission' in errors and 'timeout_answer_option' in errors
