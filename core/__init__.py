# core package
from core.data_loader import load_csv, get_summary, detect_target_columns
from core.preprocessing import preprocess_data, PreprocessingResult
from core.training import train_model, train_all_models, get_available_models
from core.evaluation import evaluate_model, compare_models
from core.persistence import save_model, load_model, list_saved_models, delete_model
