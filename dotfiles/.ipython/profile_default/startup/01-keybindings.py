from IPython import get_ipython
from prompt_toolkit.filters import ViSelectionMode, ViNavigationMode, ViMode

"""
This is a temporary (I hope) workaround for the brokenness of the first version
of vi-mode editing in prompt_toolkit.  See this issue for details:

    https://github.com/jonathanslenders/python-prompt-toolkit/issues/431
"""

ip = get_ipython()
mode = (ViNavigationMode() | ViSelectionMode()) & ViMode()

if getattr(ip, 'pt_cli'):
    registry = ip.pt_cli.application.key_bindings_registry
    is_search = lambda b: b.filter == mode
    forward = filter(is_search, registry.get_bindings_for_keys('/'))
    backward = filter(is_search, registry.get_bindings_for_keys('?'))

    for b in backward:
        registry.remove_binding(b.handler)
        registry.add_binding(
            '/',
            filter=mode,
            eager=b.eager,
            save_before=b.save_before)(b.handler)

    for b in forward:
        registry.remove_binding(b.handler)
        registry.add_binding(
            '?',
            filter=mode,
            eager=b.eager,
            save_before=b.save_before)(b.handler)
