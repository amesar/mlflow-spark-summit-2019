
"""
Dump utilities.
"""

from __future__ import print_function
import time

INDENT_INC = "  "
MAX_LEVEL = 1

def dump_run(run):
    print("Run {}".format(run.info.run_uuid))
    for k,v in run.info.__dict__.items(): print("  {}: {}".format(k[1:],v))
    print("  Params:")
    for e in run.data.params:
        print("    {}: {}".format(e.key,e.value))
    print("  Metrics:")
    for e in run.data.metrics:
        sdt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e.timestamp/1000))
        print("    {}: {}  - timestamp: {} {}".format(e.key,e.value,e.timestamp,sdt))
    print("  Tags:")
    for e in run.data.tags:
        print("    {}: {}".format(e.key,e.value))

def dump_run_info(info):
    print("Run run_uuid {}".format(info.run_uuid))
    for k,v in info.__dict__.items(): print("  {}: {}".format(k,v))

def dump_artifact(art,indent="",level=0):
    print("{}Artifact - level {}:".format(indent,level))
    for k,v in art.__dict__.items(): print("  {}{}: {}".format(indent,k[1:],v))

def _dump_artifacts(client, run_id, path, indent, level, max_level):
    level += 1
    if level > max_level: return
    artifacts = client.list_artifacts(run_id,path)
    for art in artifacts:
        dump_artifact(art,indent+INDENT_INC,level)
        if art.is_dir:
            _dump_artifacts(client, run_id, art.path, indent+INDENT_INC,level,max_level)

def dump_artifacts(client, run_id, path="", indent="", max_level=MAX_LEVEL):
    print("{}Artifacts:".format(indent))
    _dump_artifacts(client, run_id, path, indent, 0, max_level)

