from endee import Endee, Precision
e = Endee()
e.set_base_url('http://host.docker.internal:8080/api/v1')
try:
    result = e.create_index(name='nexus_knowledge_base', dimension=384, space_type='cosine', precision=Precision.INT8, sparse_model='endee_bm25')
    print('Created:', result)
except Exception as ex:
    print('Create error:', type(ex).__name__, ex)
try:
    idx = e.get_index('nexus_knowledge_base')
    print('Got index OK')
except Exception as ex:
    print('Get error:', type(ex).__name__, ex)
