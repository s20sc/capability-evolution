"""Two fixed continuations from the same complete parent; no evaluation."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from common import gate, read, write, sha, select_evidence, output_bytes
from rehearsal import teacher_targets, fit_probe, reserve_samples


def main() -> None:
    root,p=gate()
    arm=sys.argv[1]
    if arm not in p['arms'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise ValueError('arm/GPU')
    out=root/arm;out.mkdir()
    status={'arm':arm,'state':'running','model_attempts':0,'optimizer_attempts':0,'updates_returned':0,'teacher_forward_attempts':0,'teacher_samples':0}
    started=time.monotonic()
    def persist():
        status['elapsed_seconds']=time.monotonic()-started;write(out/'status.json',status)
    def check():
        if time.monotonic()-started>165: raise TimeoutError('training deadline')
        if output_bytes(root)>p['budget']['output_polling_bytes']:
            raise RuntimeError('whole output threshold')
    persist()
    try:
        sample=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        mem,util=[int(x.strip()) for x in sample.split(',')];status['GPU_admission']=sample;persist()
        if not (0<=mem<2048 and 0<=util<=5): raise RuntimeError('GPU busy; no retry')
        proposal=read(root/'proposal.json')
        for k in ['parent_evidence','acquisition_evidence','pack']:
            v=proposal[k]
            if sha(Path(v['path']))!=v['sha256']:raise ValueError('proposal input changed')
        selected=select_evidence(proposal,read(Path(proposal['parent_evidence']['path'])),read(Path(proposal['acquisition_evidence']['path'])))
        sys.path.insert(0,p['vendor'])
        import numpy as np
        import torch
        from robomimic.config import config_factory
        from robomimic.algo import algo_factory
        import robomimic.utils.obs_utils as ObsUtils
        import robomimic
        import baseline_core as core
        core.np,core.torch=np,torch
        if Path(robomimic.__file__).resolve()!=Path(p['vendor'])/'robomimic/__init__.py':raise ValueError('vendor')
        for name,value in p['official_sources'].items():
            if sha(Path(p['vendor'])/'robomimic'/name)!=value:raise ValueError('official source changed')
        expected=p['expected_runtime']
        if (torch.__version__!=expected['torch'] or np.__version__!=expected['numpy'] or torch.version.cuda!=expected['cuda']
                or torch.cuda.get_device_name(0)!=expected['gpu']):raise ValueError('runtime changed')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.06,0)
        device=torch.device('cuda:0')
        old=core.Windows(Path(p['original_data']['path']),p['original_data']['sha256'],160,24024)
        if old.ids.tolist()!=p['old_train_ids']:raise ValueError('old learning split')
        new=core.Windows(Path(proposal['pack']['path']),proposal['pack']['sha256'],3,480)
        indices=[i for i,s in enumerate(new.ids.tolist()) if s in selected]
        if indices != [2] or new.offsets.tolist()!=[0,135,311,480]:raise ValueError('selected episode span')
        cfg=config_factory('bc',dic=p['config']);ObsUtils.initialize_obs_utils_with_config(cfg)
        status['model_attempts']+=1;persist()
        model=algo_factory('bc',cfg,{k:(hi-lo,) for k,(lo,hi) in core.KEYS.items()},7,device)
        saved=torch.load(p['parent']['path'],map_location='cpu',weights_only=False)
        binding=copy.deepcopy(saved['binding'])
        if binding['config']!=p['config'] or binding['protocol_sha256']!=p['training_protocol']['sha256'] or saved['counters']['updates']!=100000:raise ValueError('complete parent metadata')
        sampler=core.Sampler(old.n,1)
        counters=core.restore(model,sampler,saved,binding)
        status['complete_restore_equal_leaves']=core.equal(saved,core.state(model,sampler,counters,binding));persist()
        del saved
        status['model_attempts']+=1;persist()
        teacher=copy.deepcopy(model.nets['policy'])
        teacher.eval();teacher.low_noise_eval=False
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        teacher_before=copy.deepcopy(teacher.state_dict())
        def fixed_probe(label):
            reserve_samples(status,4000,p['budget']['teacher_sample_vectors'])
            status.setdefault('probe_reservations',{})[label]={'teacher_forwards':1,'child_forwards':2,'sample_vectors':4000,'returned':False}
            persist()
            value=fit_probe(model,teacher,old,new,np,torch,device,p['teacher_probe_seed'])
            status['probe_reservations'][label]['returned']=True;persist()
            return value
        status['fit_before']=fixed_probe('before')
        fresh=np.random.default_rng(p['new_sampler_seed'])
        evolution={'protocol_sha256':sha(root/'protocol.json'),'arm':arm,'parent_sha256':p['parent']['sha256'],
                   'proposal_sha256':p['proposal_sha256'],'selected_ids':selected if arm=='parent_rehearsal' else [],
                   'actual_training_data':[p['original_data']['sha256']]+([proposal['pack']['sha256']] if arm=='parent_rehearsal' else []),
                   'old_windows':p['arms'][arm]['old_windows'],'new_windows':p['arms'][arm]['new_windows'],
                   'learning_operation':'parent learned-distribution rehearsal, no hard KL guarantee',
                   'teacher_parent_sha256':p['parent']['sha256'],'teacher_seed_base':p['teacher_seed_base'],
                   'teacher_low_noise_eval':False,'teacher_targets_clipped':False}
        binding['evolution']=evolution
        with (out/'updates.jsonl').open('x') as log:
            for step in range(1,p['updates_per_arm']+1):
                check();model.set_train()
                old_starts=sampler.next();n=p['arms'][arm]['old_windows']
                batch=old.batch(old_starts[:n],device);new_starts=[]
                status['teacher_forward_attempts']+=1
                reserve_samples(status,500,p['budget']['teacher_sample_vectors']);persist()
                pseudo,teacher_logp=teacher_targets(teacher,batch['obs'],p['teacher_seed_base']+step,torch,[0])
                if pseudo.shape!=batch['actions'].shape or not torch.isfinite(pseudo).all():raise ValueError('teacher target shape/finite')
                status['teacher_samples']+=pseudo.shape[0]*pseudo.shape[1]
                batch['actions']=pseudo
                if arm=='parent_rehearsal':
                    new_starts=(fresh.integers(0,169,size=50)+311).tolist()
                    added=new.batch(np.asarray(new_starts,dtype=np.int64),device)
                    batch={'obs':{k:torch.cat([batch['obs'][k],added['obs'][k]]) for k in batch['obs']},'actions':torch.cat([batch['actions'],added['actions']])}
                status['optimizer_attempts']+=1;persist()
                result=model.train_on_batch(model.process_batch_for_training(batch),epoch=counters['updates']//100+1)
                status['updates_returned']+=1
                loss=float(result['losses']['action_loss'])
                logp=result['predictions']['log_probs']
                if tuple(logp.shape)!=(100,10):raise ValueError('prediction axes')
                old_ce=-float(logp[:50].mean());new_nll=-float(logp[50:].mean())
                sampled_kl=float(teacher_logp.mean()-logp[:50].mean())
                if not all(math.isfinite(v) for v in [old_ce,new_nll,sampled_kl]):raise ValueError('nonfinite separated loss')
                if abs(loss-.5*(old_ce+new_nll))>1e-4*(1+abs(loss)):raise ValueError('loss reduction mismatch')
                if not math.isfinite(loss):raise ValueError('nonfinite learning loss')
                counters['updates']+=1;counters['sampled_windows']+=100
                if counters['updates']%100==0:model.on_epoch_end(counters['updates']//100);counters['epochs']+=1
                log.write(json.dumps({'local_update':step,'old_starts':old_starts[:n].tolist(),'new_starts':new_starts,'loss':loss,'old_teacher_ce':old_ce,'new_nll':new_nll,'sampled_old_kl':sampled_kl,'teacher_seed':p['teacher_seed_base']+step})+'\n');log.flush();persist()
        torch.cuda.synchronize();check()
        if counters['updates']!=101000 or status['updates_returned']!=1000:raise ValueError('incomplete learning')
        status['teacher_unchanged_leaves']=core.equal(teacher_before,teacher.state_dict())
        if any(q.requires_grad or q.grad is not None for q in teacher.parameters()):raise ValueError('teacher gradient mutation')
        if status['teacher_forward_attempts']!=1000 or status['teacher_samples']!=500000:raise ValueError('teacher budget')
        status['fit_after']=fixed_probe('after')
        value=core.state(model,sampler,counters,binding)
        value['evolution_sampler']=copy.deepcopy(fresh.bit_generator.state) if arm=='parent_rehearsal' else None
        checkpoint=out/'child.pt';core.save(checkpoint,value)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        status['saved_equal_leaves']=core.equal(value,loaded)
        # Restore the child into the existing model; no extra constructor or learning.
        child_counters=core.restore(model,sampler,loaded,binding)
        if arm=='parent_rehearsal':fresh.bit_generator.state=loaded['evolution_sampler']
        restored=core.state(model,sampler,child_counters,binding)
        restored['evolution_sampler']=copy.deepcopy(fresh.bit_generator.state) if arm=='parent_rehearsal' else None
        status['child_restore_equal_leaves']=core.equal(value,restored)
        if torch.cuda.max_memory_allocated(0)>805306368:raise RuntimeError('GPU allocated peak')
        status.update(state='completed',checkpoint={'path':str(checkpoint),'sha256':sha(checkpoint),'updates':101000,'evolution':evolution},peak_cuda_bytes=torch.cuda.max_memory_allocated(0));persist()
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));persist();raise


if __name__=='__main__':main()
