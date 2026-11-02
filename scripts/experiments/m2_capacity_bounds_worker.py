"""Bounded RTL simulation of both capacity consumers, not memory/abort proof."""
import re
from srnn_resident_server_worker import ROOT, VERILATOR, stage, save, stamp


def main():
    record=dict(status='starting',started_utc=stamp())
    try:
        sources=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'rtl').rglob('*.sv'))
        cmd=[VERILATOR,'--binary','--timing','--assert','-Wno-fatal','--top-module','tb_m2_capacity_bounds',
             '--Mdir','obj','-j','2']+sources+['tb_m2_capacity_bounds.sv']
        stage(record,'build',cmd,600)
        stage(record,'run',['./obj/Vtb_m2_capacity_bounds'],60)
        build=(ROOT/'build.log').read_text()
        log=(ROOT/'run.log').read_text()
        if re.search(r'%Error|%Fatal|%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)|Assertion failed',build+log):
            raise ValueError('diagnostic failure')
        if sorted(re.findall(r'^CAPACITY_PUBLIC_PASS depth=(\d+) cases=(\d+)$',log,re.M))!=[('512','16'),('578','16')]:
            raise ValueError('missing capacity cases')
        if re.findall(r'^CAPACITY_BOUNDS_PASS depths=2 cases=32$',log,re.M)!=['CAPACITY_BOUNDS_PASS depths=2 cases=32']:
            raise ValueError('missing terminal result')
        record.update(status='PASS_public_capacity_bounds_32_cases',finished_utc=stamp())
    except Exception as error:
        record.update(status='FAIL',error=repr(error),finished_utc=stamp())
        raise
    finally:
        save(record)


if __name__=='__main__':
    main()
