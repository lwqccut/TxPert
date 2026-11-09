"""Tool schemas for the local coding agent; compatible with JSON and OpenAI tools."""

def tool(name, description, properties=None, required=None):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties or {},
                           'required': required or [], 'additionalProperties': False}}}


TOOLS = [
    tool('list_files', 'List staged program modules, read-only OPAL inbox/reference files, and check artifacts.'),
    tool('read_file', 'Read a file in numbered line chunks. Only listed paths are accessible.',
         {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1},
          'max_lines': {'type': 'integer', 'minimum': 1, 'maximum': 400}}, ['path']),
    tool('write_files', 'Stage complete source files, without altering live programs. Supports analysis.py, updater.py and helper_*.py. Editing invalidates prior tests.',
         {'files': {'type': 'object', 'additionalProperties': {'type': 'string'}}}, ['files']),
    tool('run_preflight', 'Compile and execute staged candidate on same-shaped synthetic CPU tensors; returns errors and measurements. Does not prove real-data improvement.'),
    tool('run_diagnostics', 'Execute a temporary analyze(obs, memory) diagnostic on synthetic CPU tensors and return its report, without changing the staged candidate. For actual training dynamics read inbox reports or publish analysis.py.',
         {'source': {'type': 'string'}}, ['source']),
    tool('finish', 'Finish the decision. submit requires a successful preflight of the exact staged files; keep discards all staged edits.',
         {'decision': {'type': 'string', 'enum': ['submit', 'keep']}, 'reason': {'type': 'string'}},
         ['decision', 'reason']),
]
