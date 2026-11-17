"""Default-off GC routed W4A8 + finalize + indexer + head-row worker."""
import os,json,pathlib
from dsk_combined_worker import Worker as CombinedWorker
from dsk_humming_worker import encode
class Worker(CombinedWorker):
    def load_model(self,*,load_dummy_weights=False):
        enabled=os.environ.get('DSK_GC_MOE','0')
        if enabled not in ('0','1'):raise ValueError('DSK_GC_MOE must be 0 or 1')
        if enabled=='1':
            from dsk_gc_adapter import install
            install()
        super().load_model(load_dummy_weights=load_dummy_weights)
        if enabled=='0':return
        layers=[]
        for name,layer in self.model_runner.get_model().named_modules():
            q=getattr(layer,'quant_method',None)
            if getattr(q,'_dsk_gc',False):
                layers.append({'name':name,'backend':'gc_moe.sm90_fp8_fp4_gc_moe','activation':'FP8 E4M3 K128 FP32 scale','weight':'MXFP4 native GC tile/relative-exponent conversion','moe_config':encode(layer.moe_config),'weights':[list(layer.w13_weight.shape),list(layer.w2_weight.shape)],'workspace_bytes':q.execution.buffer.buffer.numel(),'output_reduced':q.moe_kernel.output_is_reduced()})
        assert len(layers)==40,len(layers)
        assert all(not hasattr(m,'humming_configs') for m in self.model_runner.get_model().modules() if getattr(getattr(m,'quant_method',None),'_dsk_gc',False))
        dest=pathlib.Path(os.environ['DSK_GC_EVIDENCE_DIR']);dest.mkdir(parents=True,exist_ok=True)
        (dest/f'rank{self.rank}.json').write_text(json.dumps({'rank':self.rank,'target_layers':layers,'target_count':40,'draft':'native unchanged','finalize':os.environ.get('DSK_GC_FINALIZE','0'),'compact_indexer':os.environ.get('DSK_COMPACT_INDEXER','0'),'headrow':os.environ.get('DSK_HEADROW','0'),'scope':'post-load configuration, not runtime kernel-hit proof'},indent=2))
