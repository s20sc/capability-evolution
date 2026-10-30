"""Native training-call accounting that preserves copied-policy dispatch."""
import copy

from contracts import LocalBudgetStop


def install(job, limits, torch):
    import robomimic.algo as algorithms
    import robosuite as suite

    original_factory = algorithms.algo_factory
    original_load = torch.load
    original_backward = torch.autograd.backward
    original_deepcopy = copy.deepcopy
    classes = set()

    def tick(name, ceiling=None):
        job.monitor()
        job.event(name)
        if ceiling is not None and job.observed[name] > ceiling:
            raise LocalBudgetStop("diagnostic operation limit: " + name)

    def model_admission(event):
        # Charge attempts before either a native factory or a policy copy.
        prior = job.observed.get("model_construction_attempts", 0) + job.observed.get("teacher_copy_attempts", 0)
        tick(event)
        if prior >= limits["models"]:
            raise LocalBudgetStop("diagnostic model budget")

    def factory(*args, **kwargs):
        model_admission("model_construction_attempts")
        model = original_factory(*args, **kwargs)
        job.event("models_constructed")
        optimizer = model.optimizers["policy"]
        original_step = optimizer.step

        def step(*a, **k):
            tick("optimizer_step_attempts", limits["optimizer_mutations"])
            job.event("optimizer_mutation_entries")
            value = original_step(*a, **k)
            job.event("optimizer_returns")
            return value

        optimizer.step = step
        cls = type(model.nets["policy"])
        if cls not in classes:
            original_forward = cls.forward_train

            def forward(policy, *a, **k):
                tick("policy_forward_attempts", limits["policy_forwards"])
                return original_forward(policy, *a, **k)

            # A class method receives the actual copied policy as self.
            # An instance closure would retain a reference to the student.
            cls.forward_train = forward
            classes.add(cls)
        return model

    def deepcopy(value, *args, **kwargs):
        teacher = type(value) in classes
        if teacher:
            model_admission("teacher_copy_attempts")
        result = original_deepcopy(value, *args, **kwargs)
        if teacher:
            job.event("teacher_copy_returns")
        return result

    def load(*args, **kwargs):
        tick("checkpoint_load_attempts", 2)
        return original_load(*args, **kwargs)

    def backward(*args, **kwargs):
        tick("gradient_attempts", limits["gradients"])
        return original_backward(*args, **kwargs)

    def no_environment(*args, **kwargs):
        tick("env_constructor_attempts", 0)

    algorithms.algo_factory = factory
    torch.load = load
    torch.autograd.backward = backward
    copy.deepcopy = deepcopy
    suite.make = no_environment
