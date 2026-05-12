# XAI Studio Phase 2 — ARCHITECTURE & INTEGRATION REVIEW

---

## ARCHITECTURE OVERVIEW

### Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (Tkinter)                       │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │   XAI View       │  │ Fairness View    │  │  Upload View     │
│  │                  │  │                  │  │                  │
│  │ • SHAP tabs      │  │ • Metric select  │  │ • File picker    │
│  │ • LIME tabs      │  │ • Alerts         │  │ • Metadata show  │
│  │ • Feature Imp    │  │ • Per-group tbl  │  │ • Validation ⚠️  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
└───────────┼──────────────────────┼──────────────────────┼──────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│              Service Layer (Business Logic)                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         PipelineService (Singleton Orchestrator)              │  │
│  │                                                               │  │
│  │  • load_data() → preprocessing_result                         │  │
│  │  • run_training() → trained_models                            │  │
│  │  • load_external_model() → loaded_model ← NEW Phase 2        │  │
│  │  • get_active_model() → returns (model, metadata) ← NEW      │  │
│  │  • get_test_set_sensitive_values() ← MISSING (needs fix)     │  │
│  │  • check_model_compatibility() ← MISSING (needs add)         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐ │
│  │   XAI Service           │  │  Visualization Service           │ │
│  │                         │  │                                  │ │
│  │ • feature_importance()  │  │ • render_confusion_matrix()      │ │
│  │ • shap_plot()           │  │ • render_roc_curve()             │ │
│  │ • lime_plot()           │  │ • render_predicted_vs_actual()   │ │
│  │ • pdp_plot()            │  │ • render_cluster_scatter()       │ │
│  │ • local_explanation()   │  │ • (15+ plot methods)             │ │
│  │                         │  │                                  │ │
│  │ Caches:                 │  │ Engine: matplotlib + plotly      │ │
│  │ • _shap_cache ⚠️ weak   │  │ (graceful fallback)              │ │
│  │ • _lime_cache ✅ OK     │  │                                  │ │
│  └─────────────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│              Core Layer (Algorithms & Models)                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Explainability (core/explainability/)          │   │
│  │                                                             │   │
│  │  ┌────────────────────┐  ┌────────────────────────────┐   │   │
│  │  │ feature_importance │  │ shap_explainer            │   │   │
│  │  │                    │  │                            │   │   │
│  │  │ • get_feature_…()  │  │ • compute_shap_values() ✅ │   │   │
│  │  │ • plot_…()         │  │ • plot_shap_summary() 🔴  │   │   │
│  │  │                    │  │ • plot_shap_waterfall() 🔴 │   │   │
│  │  │ Issues: ⚠️ edge    │  │ • plot_shap_force()       │   │   │
│  │  │   cases, no bounds │  │                            │   │   │
│  │  │                    │  │ Issues: 🔴 multiclass,     │   │   │
│  │  └────────────────────┘  │ KernelExplainer hang      │   │   │
│  │  ┌────────────────────┐  └────────────────────────────┘   │   │
│  │  │ lime_explainer     │  ┌────────────────────────────┐   │   │
│  │  │                    │  │ pdp_explainer             │   │   │
│  │  │ • create_…()       │  │                            │   │   │
│  │  │ • explain_…()  ⚠️  │  │ • compute_pdp_1d()        │   │   │
│  │  │ • plot_lime_…() ✅ │  │ • compute_pdp_2d()        │   │   │
│  │  │                    │  │                            │   │   │
│  │  │ Issues: ⚠️ predict_proba │ Issues: None ✅       │   │   │
│  │  │   fallback         │  │                            │   │   │
│  │  └────────────────────┘  └────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Fairness (core/fairness/)                      │   │
│  │                                                             │   │
│  │  • compute_fairness_metrics() 🔴 edge cases               │   │
│  │  • plot_fairness_comparison() ✅                           │   │
│  │                                                             │   │
│  │  Issues: 🔴 division by zero semantics,                    │   │
│  │         di_ratio edge case, single group not blocked       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          Persistence (core/persistence.py)                 │   │
│  │                                                             │   │
│  │  • save_model() — saves {model, metadata} dict ✅         │   │
│  │  • load_model() — restores model + metadata ✅             │   │
│  │  • list_saved_models() ✅                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │     Model Loading (core/model_loader.py)                   │   │
│  │                                                             │   │
│  │  • load_model_file() — supports .pkl/.joblib ✅           │   │
│  │  • detect_model_info() — extracts metadata ✅             │   │
│  │                                                             │   │
│  │  Issues: None ✅                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                    Data Layer                                       │
│  • Pandas DataFrames (in-memory)                                   │
│  • Numpy arrays (feature matrices)                                 │
│  • Pickle/Joblib models (persisted)                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## DATA FLOW DIAGRAMS

### Phase 1: Model Training Pipeline (Existing)
```
User Input (CSV)
    ↓
load_data()
    ↓ (DataFrame)
run_preprocessing()
    ↓ (PreprocessingResult: X_train, y_train, X_test, y_test, feature_names)
run_training()
    ↓ (dict of {model_name: {model, training_time, error}})
run_evaluation()
    ↓ (dict of {model_name: {metrics, confusion_matrix, ...}})
save_trained_model()
    ↓ (Serialized to .pkl with metadata)
Disk Storage
```

### Phase 2: Model Upload + XAI Pipeline (New)
```
External Model File (.pkl)
    ↓
load_external_model()
    ├─→ load_model_file()          (Extract model + metadata)
    ├─→ detect_model_info()        (Enrich metadata)
    └─→ loaded_model (in PipelineService)
    
User selects XAI method
    ↓
get_active_model()               (Returns loaded_model or first trained_model)
    ↓ (model, metadata)
XAIService.shap_plot()          (or lime_plot, feature_importance, pdp_plot)
    ├─→ compute_shap_values()    (Uses service caching)
    ├─→ plot_shap_summary()
    └─→ PlotCanvas.update_figure()
    
User selects fairness analysis
    ↓
get_active_model()
    ↓ (model)
get_test_set_sensitive_values()  ← MISSING (needs implementation)
    ↓ (sensitive_values aligned with y_test)
compute_fairness_metrics()
    ├─→ metrics per group
    ├─→ alerts
    └─→ visualization
```

### Data Alignment Issue (Currently Fragile)
```
Original DataFrame (1000 rows)
    ↓ [drop rows with null target]
Valid rows (990 rows)
    ↓ [train_test_split(random_state=42)]
    ├─→ Train indices (790 rows) → X_train, y_train
    └─→ Test indices (200 rows)  → X_test, y_test
    
Fairness Analysis:
    ↓ get predictions on X_test
    ↓ y_pred (200 values)
    ↓ y_true (200 values)
    ⚠️ Need: sensitive_values (200 values, aligned with indices)
    
Current approach: Reconstruct indices by re-doing train_test_split
    ← FRAGILE: assumes same random_state, test_size, stratification
    
Proposed approach: Store indices in PreprocessingResult during preprocessing
    ← ROBUST: actual indices available
```

---

## INTEGRATION POINTS & DEPENDENCIES

### 1. PipelineService → XAIService
```
PipelineService.preprocessing_result
    ├─ X_test                    → Used by: feature_importance, shap_plot, lime_plot, pdp_plot
    ├─ y_test                    → Used by: feature_importance (permutation), local_explanation
    ├─ feature_names             → Used by: all explainability methods
    ├─ task_type                 → Used by: lime_plot (classification/regression mode)
    └─ X_train                   → Used by: lime_plot (explainer training), shap (background)

PipelineService.trained_models
    ├─ model                     → Used by: get_active_model()
    ├─ training_time             → Used by: metadata
    └─ metadata                  → Used by: feature_names, task_type

PipelineService.loaded_model (NEW Phase 2)
    ├─ model                     → Used by: get_active_model() [priority over trained]
    └─ metadata                  → Used by: feature_names, task_type
```

### 2. XAIService → Core Explainability
```
XAIService.feature_importance()
    → core.explainability.feature_importance.get_feature_importance()
    → core.explainability.feature_importance.plot_feature_importance()

XAIService.shap_plot()
    → core.explainability.shap_explainer.compute_shap_values()
    → core.explainability.shap_explainer.plot_shap_*()
    Cache: {model_id, X_data_shape} → shap_values ⚠️ WEAK

XAIService.lime_plot()
    → core.explainability.lime_explainer.create_lime_explainer()
    → core.explainability.lime_explainer.explain_instance()
    → core.explainability.lime_explainer.plot_lime_explanation()
    Cache: {model_id, X_train_shape} → explainer ✅ GOOD

XAIService.pdp_plot()
    → core.explainability.pdp_explainer.compute_pdp_1d()
    → core.explainability.pdp_explainer.compute_pdp_2d()
```

### 3. FairnessView → Core Fairness + Pipeline
```
Current (FRAGILE):
    fairness_view._run_analysis()
        ├─ get_active_model()         [OK]
        ├─ model.predict(X_test)      [OK]
        ├─ Reconstruct test indices   [🔴 FRAGILE]
        │   └─ train_test_split(random_state, test_size)
        │   └─ df[sensitive_col].values[test_indices]
        └─ compute_fairness_metrics() [OK]

Proposed (ROBUST):
    fairness_view._run_analysis()
        ├─ get_active_model()         [OK]
        ├─ model.predict(X_test)      [OK]
        ├─ get_test_set_sensitive_values(sensitive_col) [NEW in pipeline_service]
        └─ compute_fairness_metrics() [OK]
```

### 4. UploadView → PipelineService + ModelLoader
```
Current:
    upload_view._on_upload()
        ├─ filedialog.askopenfilename()
        ├─ pipeline_service.load_external_model(filepath)
        │   ├─ core.model_loader.load_model_file()
        │   ├─ core.model_loader.detect_model_info()
        │   └─ Store in loaded_model ✅
        └─ display_model_info()       [No validation ⚠️]

Issues:
    ⚠️ No model ↔ preprocessing compatibility check
    ⚠️ No feature count validation
    ⚠️ Synchronous loading (UI freezes on large models)
    ⚠️ No task type mismatch warning

Proposed additions:
    ├─ File size pre-check (max 500MB)
    ├─ Async loading with thread
    ├─ pipeline_service.check_model_compatibility()
    └─ Block if incompatible
```

---

## SINGLETON PATTERN USAGE

### Current State
```
✅ PipelineService
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    Usage:
        service = PipelineService()  # Always same instance
        service.dataframe            # Shared state
        service.trained_models       # Shared state

⚠️ XAIService (NOT singleton)
    service = XAIService()           # NEW instance each time
    service._shap_cache             # Local cache, not shared
    
    Problem:
        • Each UI tab creates new XAIService()
        • SHAP cache not reused between tabs
        • LIME cache not reused between tabs

✅ VisualizationService (NOT singleton, but passed around)
    viz = VisualizationService()
    
    Usage in XAIService:
        def __init__(self, viz: VisualizationService | None = None):
            self._viz = viz or VisualizationService()
    
    This is OK because it's stateless
```

### Recommendation
Make XAIService a singleton to share caches:
```python
class XAIService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._viz = VisualizationService()
        self._shap_cache = {}
        self._lime_cache = {}
```

Then use:
```python
# In xai_view.py
self._xai_service = XAIService()  # Singleton, caches persist
```

---

## CACHING STRATEGY REVIEW

### Current Approach

| Component | Cache Key | Cached Value | Issue | TTL |
|-----------|-----------|--------------|-------|-----|
| SHAP | `f"{id(model)}-{X_data.shape}"` | shap_values | 🔴 Weak (no data hash) | ∞ |
| LIME | `f"{id(model)}-{X_train.shape}"` | explainer | ✅ Good (data rarely changes) | ∞ |
| Feature Importance | None | — | ⚠️ Not cached | — |
| PDP | None | — | ⚠️ Not cached | — |

### Proposed Improvements

```python
# 1. Better SHAP cache key (with data content hash)
cache_key = f"{type(model).__name__}-{X_data.shape}-{hash_rows(X_data)}"

# 2. Add Feature Importance caching
class XAIService:
    def __init__(self):
        self._fi_cache = {}  # {model_id: feature_importances}

# 3. Add cache size limits
MAX_CACHE_ENTRIES = 5  # Keep recent 5 computations
self._cache = OrderedDict()
if len(self._cache) > MAX_CACHE_ENTRIES:
    self._cache.popitem(last=False)  # Remove oldest

# 4. Add cache invalidation on data change
# When preprocessing_result changes, clear caches:
def run_preprocessing(self, ...):
    # ... preprocessing ...
    self.xai_service._clear_caches()  # Invalidate caches
```

---

## TYPE HINTS AUDIT

### Current State

| File | Type Hints Coverage | Issues |
|------|-------------------|--------|
| feature_importance.py | None | 0% coverage |
| lime_explainer.py | None | 0% coverage |
| shap_explainer.py | None | 0% coverage |
| bias_detector.py | None | 0% coverage |
| xai_service.py | Partial (LocalExplanation) | 40% coverage |
| visualization_service.py | Partial (PlotPayload) | 30% coverage |
| xai_view.py | None | 0% coverage |
| fairness_view.py | None | 0% coverage |
| upload_view.py | None | 0% coverage |
| **Overall** | **Low** | **~10% coverage** |

### Recommended Type Hints (Priority Order)

```python
# 1. Core explainability modules (highest impact)
def get_feature_importance(
    model: Any,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Extract and plot feature importance from a model."""

def compute_shap_values(
    model: Any,
    X_data: np.ndarray,
    feature_names: list[str] | None = None,
) -> Any:  # Returns shap.Explanation
    """Compute SHAP values for model and data."""

# 2. Service layer
def shap_plot(
    self,
    model: Any,
    X_data: np.ndarray,
    feature_names: list[str],
    plot_type: str = "bar",
    instance_idx: int = 0,
) -> plt.Figure:
    """Generate SHAP plot."""

# 3. UI views
def _run_shap(self) -> None:
    """Compute and display SHAP plot."""
```

---

## ERROR HANDLING CONSISTENCY

### Current State

| Module | Try/Except | Logging | User Feedback |
|--------|-----------|---------|--------------|
| SHAP explainer | ⚠️ Silent fallback | ⚠️ Partial | ❌ No UI warning |
| LIME explainer | ✅ Raises | ✅ Good | ⚠️ Generic error |
| Feature importance | ⚠️ Silent fallback | ✅ Good | — |
| Fairness detector | ❌ None | ✅ Good | — |
| XAI View | ✅ Raises, catches | ✅ Good | ✅ Error dialog |
| Fairness View | ✅ Raises, catches | ✅ Good | ✅ Error dialog |

### Issues

```python
# Problem 1: Silent fallback in SHAP (line 23-34)
try:
    return shap.TreeExplainer(model)
except Exception:  # ← Silently swallows ALL exceptions
    logger.warning("TreeExplainer failed, falling back.")

# Better:
except Exception as e:
    logger.warning(f"TreeExplainer failed ({e.__class__.__name__}): {e}")

# Problem 2: No exception type specificity
except Exception:  # ← Catches KeyboardInterrupt, SystemExit, etc.

# Better:
except (ValueError, AttributeError, RuntimeError) as e:
    logger.warning(...)

# Problem 3: Async exceptions not caught
def _run_shap(self):
    threading.Thread(target=_compute, daemon=True).start()
    # If _compute raises, exception is lost!

# Better:
def _compute():
    try:
        # ... computation ...
    except Exception as e:
        logger.error("SHAP computation failed: %s", e)
        self._shap_canvas.after(
            0,
            lambda: show_error(_("xai_err_shap"), str(e))
        )
```

---

## THREAD SAFETY & CONCURRENCY

### Current Usage

```python
# Async computations in UI views (Tkinter)
threading.Thread(target=_compute, daemon=True).start()

# Issues:
1. Daemon threads can be terminated abruptly
2. No thread pool (unbounded thread creation)
3. Shared state access (PipelineService) without locks
   - But Tkinter runs in main thread only
   - background threads only read, don't modify
   - So likely OK, but fragile

# Safe because:
- Threads only call model.predict(), explainer.explain()
- Don't modify PipelineService state
- Results posted back to main thread via self.after()
```

### Recommendation
```python
# Use thread pool for bounded concurrency
from concurrent.futures import ThreadPoolExecutor

class XAIService:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def compute_shap_async(self, model, X_data, callback):
        future = self._executor.submit(
            compute_shap_values, model, X_data
        )
        future.add_done_callback(callback)
```

---

## PERFORMANCE CHARACTERISTICS

| Operation | Time | Notes | Optimization |
|-----------|------|-------|--------------|
| Tree SHAP | 0.1-1s | Fast for small trees | OK |
| Linear SHAP | 1-5s | Medium | OK |
| Kernel SHAP (100 bg, 10k samples) | 5-60m | **VERY SLOW** | ❌ Use approximation or limit |
| LIME explanation | 2-10s | Medium | Batch multiple |
| Permutation importance (10 repeats) | 5-30s | Medium | ⚠️ Configurable repeats |
| PDP (1000 grid points) | 1-5s | OK | OK |
| Fairness metrics (10k samples) | <1s | Fast | OK |

### Bottlenecks

```python
# 1. SHAP KernelExplainer (inherently slow)
# O(n_background * n_samples * n_features^2)
# Solution: Subsample background, cap n_samples

# 2. Permutation importance (10 repeats)
# Solution: Make configurable (5 for quick check, 20 for accurate)

# 3. PDP grid resolution (50 points default)
# Solution: Add UI control, default to 30
```

---

## SUMMARY: INTEGRATION HEALTH

| Aspect | Status | Score | Issues |
|--------|--------|-------|--------|
| **Layer Separation** | ✅ Good | 8/10 | Business logic in UI (fairness view) |
| **State Management** | ⚠️ Fragile | 6/10 | PipelineService singleton OK, XAIService not singleton, index reconstruction complex |
| **Error Handling** | ⚠️ Inconsistent | 6/10 | SHAP silent fallback, no exception type specificity, async errors not caught |
| **Type Safety** | ❌ Poor | 2/10 | <10% type hint coverage |
| **Caching** | ⚠️ Weak | 5/10 | SHAP cache key fragile, LIME OK, FI/PDP not cached |
| **Thread Safety** | ✅ OK | 7/10 | Mostly safe by design, but no explicit synchronization |
| **Performance** | ⚠️ Adequate | 6/10 | SHAP KernelExplainer too slow, no timeouts |
| **Documentation** | ⚠️ Minimal | 4/10 | Few docstrings, no architecture docs |

**Overall Integration Score: 6/10**

---

## PHASE 2→3 MIGRATION CHECKLIST

### Before Merging to Main
- [ ] All 6 critical issues fixed and tested
- [ ] Type hints added to core explainability modules
- [ ] Integration tests cover cross-component flows
- [ ] Model compatibility validation in upload
- [ ] Business logic moved from UI to service layer
- [ ] Caching strategy documented

### Phase 3 Goals
- [ ] Make XAIService a singleton
- [ ] Add caching for Feature Importance and PDP
- [ ] Implement SHAP computation timeout
- [ ] Add async model loading
- [ ] Full type hint coverage
- [ ] Comprehensive error handling consistency
