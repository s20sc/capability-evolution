"""Fixed evidence intersection and eligible-demonstration selection."""

def select_evidence(proposal: dict, parent: dict, acquisition: dict) -> list[str]:
    if parent['status'] != 'completed' or acquisition['status'] != 'completed':
        raise ValueError('incomplete evidence')
    if len(parent['cases']) != 4 or len(acquisition['cases']) != 4: raise ValueError('four input records required')
    pc = {c['seed']: c for c in parent['cases']}
    ac = {c['seed']: c for c in acquisition['cases']}
    if len(pc)!=4 or len(ac)!=4 or set(pc)!=set(ac): raise ValueError('four unique paired scenes')
    selected=[]
    for seed,c in pc.items():
        g=ac[seed]
        if c['initial_state_sha256'] != g['initial_state_sha256']: raise ValueError('initial mismatch')
        if not c['success'] and c['stop_reason']=='horizon' and g['training_eligible']:
            selected.append('generated_'+str(seed))
    if selected != proposal['selected_demo_ids'] or selected != ['generated_847811185']:
        raise ValueError('fixed feedback selection changed')
    return selected

def all_valid_ids(acquisition):
    if acquisition['status']!='completed' or len(acquisition['cases'])!=4:
        raise ValueError('complete collection required')
    cases=acquisition['cases']
    if len({c['seed'] for c in cases})!=4:
        raise ValueError('unique collection conditions')
    if any(type(c['training_eligible']) is not bool for c in cases):
        raise ValueError('boolean eligibility required')
    return ['generated_'+str(c['seed']) for c in cases if c['training_eligible']]
