"""Five independently restored D continuations, fixed window budgets."""
import copy
import math
import os
from pathlib import Path
import sys
import time
from common import gate,read,write,sha,output_bytes
from reuse import check_gpu_idle


def main() -> None:
    root,p=gate();arm=sys.argv[1]
    if arm not in p['arms'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise ValueError('arm/device')
    out=root/arm;out.mkdir()
    status={'arm':arm,'state':'running','model_attempts':0,'optimizer_attempts':0,'updates_returned':0}
    started=time.monotonic()
    def persist():
        status['elapsed_seconds']=time.monotonic()-started;write(out/'status.json',status)
    persist()
    try:
        check_gpu_idle(status,persist)
        sys.path.insert(0,p['vendor'])
        import numpy as np
        import torch
        from robomimic.config import config_factory
        from robomimic.algo import algo_factory
        import robomimic.utils.obs_utils as ObsUtils
        import baseline_core as core
        core.np,core.torch=np,torch
        if torch.__version__!='2.7.0+cu128' or np.__version__!='1.26.0' or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':
            raise RuntimeError('frozen runtime')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.06,0)
        device=torch.device('cuda:0')
        old=core.Windows(Path(p['original_data']['path']),p['original_data']['sha256'],160,24024)
        target=core.Windows(Path(p['target_data']['path']),p['target_data']['sha256'],3,480)
        if old.ids.tolist()!=p['old_train_ids'] or target.offsets.tolist()!=[0,135,311,480]: raise ValueError('learning split/target spans')
        correction=None
        if arm!='C0':
            info=read(root/'packs.json')[arm]
            correction=core.Windows(Path(info['path']),info['sha256'],3,info['rows'])
            if correction.ids.tolist()!=[str(x) for x in p['arms'][arm]['seeds']]:raise ValueError('correction membership/order')
        cfg=config_factory('bc',dic=p['config']);ObsUtils.initialize_obs_utils_with_config(cfg)
        status['model_attempts']+=1;persist()
        model=algo_factory('bc',cfg,{k:(hi-lo,) for k,(lo,hi) in core.KEYS.items()},7,device)
        saved=torch.load(p['parent']['path'],map_location='cpu',weights_only=False)
        binding=copy.deepcopy(saved['binding'])
        if binding['config']!=p['config'] or saved['counters']['updates']!=101000:raise ValueError('D identity')
        sampler=core.Sampler(old.n,1);target_rng=np.random.default_rng(1)
        counters=core.restore(model,sampler,saved,binding)
        target_rng.bit_generator.state=copy.deepcopy(saved['evolution_sampler'])
        restored=core.state(model,sampler,counters,binding);restored['evolution_sampler']=copy.deepcopy(target_rng.bit_generator.state)
        status['complete_parent_restore_leaves']=core.equal(saved,restored);persist()
        del saved,restored
        correction_rng=np.random.default_rng(p['correction_sampler_seed'])
        evolution={'protocol_sha256':sha(root/'protocol.json'),'arm':arm,'parent_sha256':p['parent']['sha256'],
                   'selection':p['arms'][arm],'correction_pack':None if arm=='C0' else read(root/'packs.json')[arm],
                   'operation':'fixed actual-action BC continuation; not original D teacher objective'}
        binding['evolution']=evolution
        with (out/'updates.jsonl').open('x') as log:
            for local_step in range(1,p['updates_per_arm']+1):
                if time.monotonic()-started>p['train_seconds']-10:raise TimeoutError('train deadline')
                if output_bytes(root)>p['output_polling_bytes']:raise RuntimeError('output polling threshold')
                model.set_train()
                old_starts=sampler.next();target_starts=target_rng.integers(0,169,size=40)+311
                base=old.batch(old_starts[:40],device);new=target.batch(target_starts,device)
                correction_starts=[]
                if correction is None:
                    extra=old.batch(old_starts[40:60],device)
                else:
                    indices=correction_rng.integers(0,3,size=20)
                    for index in indices:
                        correction_starts.append(int(correction_rng.integers(correction.offsets[index],correction.offsets[index+1])))
                    extra=correction.batch(np.asarray(correction_starts,dtype=np.int64),device)
                batch={'obs':{k:torch.cat([base['obs'][k],new['obs'][k],extra['obs'][k]]) for k in base['obs']},
                       'actions':torch.cat([base['actions'],new['actions'],extra['actions']])}
                status['optimizer_attempts']+=1;persist()
                result=model.train_on_batch(model.process_batch_for_training(batch),epoch=counters['updates']//100+1)
                status['updates_returned']+=1
                loss=float(result['losses']['action_loss'])
                if not math.isfinite(loss):raise ValueError('nonfinite loss')
                counters['updates']+=1;counters['sampled_windows']+=100
                if counters['updates']%100==0:model.on_epoch_end(counters['updates']//100);counters['epochs']+=1
                log.write(__import__('json').dumps({'local_step':local_step,'old_starts':old_starts[:60 if arm=='C0' else 40].tolist(),
                    'target_starts':target_starts.tolist(),'correction_starts':correction_starts,'loss':loss})+'\n');log.flush();persist()
        torch.cuda.synchronize()
        if counters['updates']!=102000 or status['updates_returned']!=1000:raise ValueError('learning count')
        value=core.state(model,sampler,counters,binding)
        value['evolution_sampler']=copy.deepcopy(target_rng.bit_generator.state)
        value['correction_sampler']=copy.deepcopy(correction_rng.bit_generator.state)
        checkpoint=out/'child.pt';core.save(checkpoint,value)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        status['saved_equal_leaves']=core.equal(value,loaded)
        counters=core.restore(model,sampler,loaded,binding)
        target_rng.bit_generator.state=loaded['evolution_sampler'];correction_rng.bit_generator.state=loaded['correction_sampler']
        restored=core.state(model,sampler,counters,binding)
        restored.update(evolution_sampler=copy.deepcopy(target_rng.bit_generator.state),correction_sampler=copy.deepcopy(correction_rng.bit_generator.state))
        status['child_restore_equal_leaves']=core.equal(value,restored)
        status.update(state='completed',checkpoint={'path':str(checkpoint),'sha256':sha(checkpoint),'updates':102000,'evolution':evolution},peak_cuda_bytes=torch.cuda.max_memory_allocated(0));persist()
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));persist();raise


if __name__=='__main__':main()
