class Executor:

    def __init__(self, root_module):
        self.root_module = root_module
        self.modules = {}
    
    def run(self, schedule):
        existing_results = {}

        for callable_ref, batch_key, step_nodes in schedule:
            #args = [[node.args[i] for node in step_nodes]
            #        for i in range(len(batch_key.args))]
            #kwargs = {
            #    k: [node.kwargs[k] for node in step_nodes]
            #    for k, shape in batch_key.kwargs}

            # Find the module to invoke
            module = self._get_module(callable_ref)

            # Invoke the module
            results = batch_key.call_batched(
                existing_results, module, callable_ref, step_nodes)

            # Save the results
            for node, result in zip(step_nodes, results):
                existing_results[node] = result
                for next_node in node.outgoing:
                    next_node.num_incoming -= 1

        return existing_results

    def _get_module(self, callable_ref):
        cached = self.modules.get(callable_ref)
        if cached is not None:
            return cached
        self.modules[callable_ref] = callable_ref.find(self.root_module)
        return result