from models.data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from datetime import datetime
print('Starting training at', datetime.today())

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device)

maml_system = ExperimentBuilder(model=model, data=MetaLearningSystemDataLoader, args=args, device=device)
maml_system.run_experiment()
# maml_system.run_test([1,2])
