"""
Task suite: T1–T6 as defined in the paper (Table 5).

Each task maps to a robosuite environment with specific configuration.
Success criteria and ECM step counts are defined per-task.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .robosuite_wrapper import RobosuiteWrapper


@dataclass
class TaskConfig:
    """Configuration for a single evaluation task."""
    name: str
    robosuite_env: str
    description: str
    success_criterion: str
    ecm_steps: int
    single_object: bool = True
    num_objects: int = 1
    extra_kwargs: Dict = field(default_factory=dict)


# Paper Table 5: six manipulation tasks of increasing complexity
DEFAULT_TASKS = [
    TaskConfig(
        name="T1_Pick",
        robosuite_env="PickPlaceSingle" if False else "PickPlace",  # use PickPlace with single obj
        description="Grasp target object from table",
        success_criterion="object lifted >5cm",
        ecm_steps=2,
        single_object=True,
    ),
    TaskConfig(
        name="T2_Place",
        robosuite_env="PickPlace",
        description="Place object at goal position",
        success_criterion="position error <2cm",
        ecm_steps=3,
        single_object=True,
    ),
    TaskConfig(
        name="T3_Stack",
        robosuite_env="Stack",
        description="Stack object A onto object B",
        success_criterion="stable stack for 3s",
        ecm_steps=4,
        single_object=False,
    ),
    TaskConfig(
        name="T4_NutAssembly",
        robosuite_env="NutAssembly",
        description="Insert nut onto peg (replaces 'Pour' — more standard benchmark)",
        success_criterion="nut seated on correct peg",
        ecm_steps=5,
        single_object=False,
    ),
    TaskConfig(
        name="T5_PickPlaceMulti",
        robosuite_env="PickPlace",
        description="Sort 3 objects by type into target bins",
        success_criterion="all 3 objects correctly placed",
        ecm_steps=6,
        single_object=False,
        num_objects=3,
    ),
    TaskConfig(
        name="T6_TwoArmPegInHole",
        robosuite_env="TwoArmPegInHole",
        description="Bimanual peg-in-hole insertion",
        success_criterion="peg fully inserted, force check passes",
        ecm_steps=8,
        single_object=False,
    ),
]


class TaskSuite:
    """
    Manages the suite of T1–T6 tasks.
    Creates robosuite-based environments on demand.
    """

    def __init__(
        self,
        tasks: Optional[List[TaskConfig]] = None,
        robot: str = "Panda",
        controller: str = "OSC_POSE",
        horizon: int = 500,
        reward_shaping: bool = True,
        object_pos_noise: float = 0.05,
        obs_noise_std: float = 0.01,
        actuation_failure_prob: float = 0.05,
        force_limit: float = 20.0,
        workspace_bounds: Optional[Dict] = None,
        forbidden_zones: Optional[list] = None,
        seed: int = 42,
    ):
        self.tasks = tasks or DEFAULT_TASKS
        self.robot = robot
        self.controller = controller
        self.horizon = horizon
        self.reward_shaping = reward_shaping
        self.object_pos_noise = object_pos_noise
        self.obs_noise_std = obs_noise_std
        self.actuation_failure_prob = actuation_failure_prob
        self.force_limit = force_limit
        self.workspace_bounds = workspace_bounds
        self.forbidden_zones = forbidden_zones
        self.seed = seed

    def make_env(self, task: TaskConfig) -> RobosuiteWrapper:
        """Create a wrapped robosuite environment for the given task."""
        robots = self.robot
        kwargs = dict(task.extra_kwargs)

        return RobosuiteWrapper(
            env_name=task.robosuite_env,
            robot=robots,
            controller=self.controller,
            horizon=self.horizon,
            reward_shaping=self.reward_shaping,
            object_pos_noise=self.object_pos_noise,
            obs_noise_std=self.obs_noise_std,
            actuation_failure_prob=self.actuation_failure_prob,
            force_limit=self.force_limit,
            workspace_bounds=self.workspace_bounds,
            forbidden_zones=self.forbidden_zones,
            seed=self.seed,
            **kwargs,
        )

    def get_task(self, name: str) -> TaskConfig:
        """Retrieve a task config by name."""
        for t in self.tasks:
            if t.name == name:
                return t
        raise ValueError(f"Unknown task: {name}")

    def all_tasks(self) -> List[TaskConfig]:
        return list(self.tasks)

    def single_ecm_tasks(self) -> List[TaskConfig]:
        """T1–T3: exercise single ECM capabilities."""
        return [t for t in self.tasks if t.ecm_steps <= 4]

    def composition_tasks(self) -> List[TaskConfig]:
        """T4–T6: require composing multiple ECMs."""
        return [t for t in self.tasks if t.ecm_steps > 4]
