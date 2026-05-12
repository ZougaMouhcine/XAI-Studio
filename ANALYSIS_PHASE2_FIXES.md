# XAI Studio Phase 2 — QUICK FIX GUIDE

**Status:** 6 Critical Issues + 21 Important Issues  
**Estimated Fix Time:** 4-6 hours for critical, 2-3 hours for important  

---

## CRITICAL ISSUES — FIX TODAY

### 1. SHAP Multiclass Crash (shap_explainer.py:139-145)

**Problem:** Assumes class 1 exists, crashes on multiclass > 2
```python
# Current (WRONG)
if sv.values.ndim > 1:
    sv = shap.Explanation(
        values=sv.values[:, 1] if sv.values.shape[1] > 1 else sv.values[:, 0],
        base_values=sv.base_values[1] if hasattr(sv.base_values, '__len__') else sv.base_values,
```

**Fix:**
```python
# Corrected
if sv.values.ndim > 1:
    # Handle multiclass: select class with highest mean absolute SHAP value
    if sv.values.shape[1] > 1:
        class_idx = np.argmax(np.abs(sv.values).mean(axis=0))
    else:
        class_idx = 0
    
    sv = shap.Explanation(
        values=sv.values[:, class_idx],
        base_values=(
            sv.base_values[class_idx] 
            if isinstance(sv.base_values, (list, np.ndarray)) and len(sv.base_values) > class_idx
            else sv.base_values
        ),
        data=sv.data,
        feature_names=feature_names,
    )
```

**Test:** 
```python
# Test with 3-class classification
model = RandomForestClassifier(n_classes=3)
model.fit(X_train, y_train_3class)
shap_values = compute_shap_values(model, X_test[:10])
fig = plot_shap_waterfall(shap_values, 0)  # Should NOT crash
```

---

### 2. SHAP KernelExplainer Hang (shap_explainer.py:45-46)

**Problem:** Can run for hours on large feature sets
```python
# Current
if len(X_background) > 100:
    bg = shap.sample(X_background, 100)
```

**Fix:**
```python
logger.warning(
    "Using SHAP KernelExplainer (slow) for %s with %d features. "
    "This will be slow. Consider using a tree-based model for faster explanations.",
    class_name, X_background.shape[1]
)

# Subsample background (max 50 to speed up)
if len(X_background) > 50:
    bg_idx = np.random.choice(len(X_background), 50, replace=False)
    bg = X_background[bg_idx]
else:
    bg = X_background

predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

# Add timeout context (requires signal on Unix, but at least warn)
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("SHAP computation exceeded 60 seconds")

try:
    # Unix only
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    explainer = shap.KernelExplainer(predict_fn, bg)
    signal.alarm(0)  # Cancel alarm
except (TimeoutError, AttributeError):  # AttributeError on Windows
    logger.error("SHAP KernelExplainer computation exceeded timeout")
    raise RuntimeError("SHAP computation too slow. Consider using tree-based model.")
```

---

### 3. Permutation Importance Result Crash (feature_importance.py:73-74)

**Problem:** Assumes `result.importances_mean` exists
```python
# Current
result = permutation_importance(...)
importances = result.importances_mean  # Crashes if attribute missing
```

**Fix:**
```python
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1,
)

# Safe extraction with fallback
if hasattr(result, 'importances_mean'):
    importances = result.importances_mean
elif hasattr(result, 'importances'):
    # Fallback: manually compute mean across repeats
    importances = np.mean(result.importances, axis=0)
else:
    raise ValueError(
        f"Unexpected permutation importance result type: {type(result)}. "
        f"Expected 'importances_mean' or 'importances' attribute."
    )

if len(importances) == 0:
    raise ValueError("Permutation importance produced empty result")

logger.info("Permutation importance computed (%s, n_features=%d)", class_name, len(importances))
```

---

### 4. Fairness Index Reconstruction Logic Error (fairness_view.py:137-152)

**Problem:** Complex, fragile index reconstruction doesn't match preprocessing

**Solution: Move to PipelineService**

Add to `services/pipeline_service.py`:
```python
def get_test_set_sensitive_values(self, sensitive_col: str, random_state: int = 42) -> np.ndarray:
    """
    Get sensitive attribute values aligned with test set.
    
    Reconstructs the test indices using the same logic as preprocessing
    to ensure alignment with y_test and y_pred.
    
    Parameters
    ----------
    sensitive_col : str
        Column name in the original dataframe
    random_state : int
        Must match the random_state used in preprocessing
    
    Returns
    -------
    np.ndarray
        Sensitive attribute values for test set (same length as y_test)
    """
    if self.dataframe is None:
        raise RuntimeError("No data loaded")
    if sensitive_col not in self.dataframe.columns:
        raise ValueError(f"Column '{sensitive_col}' not found in dataset")
    if self.preprocessing_result is None:
        raise RuntimeError("Data not preprocessed yet")
    
    df = self.dataframe
    pr = self.preprocessing_result
    
    # Get target column and build valid mask
    target_col = self.target_columns[0] if self.target_columns else None
    if target_col and target_col in df.columns:
        valid_mask = df[target_col].notnull()
    else:
        valid_mask = np.ones(len(df), dtype=bool)
    
    valid_indices = np.where(valid_mask)[0]
    
    # Reconstruct test split using same random state
    from sklearn.model_selection import train_test_split
    test_size = len(pr.X_test) / len(valid_indices)
    _, test_indices = train_test_split(
        valid_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col].values[valid_indices] if target_col else None,
    )
    
    # Get sensitive values aligned with test set
    sensitive_values = df[sensitive_col].values[test_indices]
    
    # Ensure same length as predictions (handle edge cases)
    sensitive_values = sensitive_values[:len(pr.X_test)]
    
    logger.info("Retrieved %d sensitive values for %s", len(sensitive_values), sensitive_col)
    return sensitive_values
```

Then update `ui/views/fairness_view.py` to use it:
```python
def _run_analysis(self):
    # ... existing validation ...
    
    def _compute():
        try:
            from core.fairness.bias_detector import (
                compute_fairness_metrics, plot_fairness_comparison,
            )
            
            y_pred = model.predict(pr.X_test)
            y_true = pr.y_test
            
            # Use service method instead of complex reconstruction
            sensitive_values = self._service.get_test_set_sensitive_values(sensitive_col)
            
            results = compute_fairness_metrics(y_true, y_pred, sensitive_values)
            fig = plot_fairness_comparison(results, metric_key)
            # ... rest unchanged
```

---

### 5. SHAP Cache Key Too Weak (xai_service.py:44)

**Problem:** Uses memory address + shape, misses data content changes

**Fix:**
```python
@staticmethod
def _cache_key(model, X_data):
    """
    Generate cache key for SHAP/LIME results.
    
    Includes model class name, data shape, AND hash of first/last rows
    to catch data changes while maintaining reasonable performance.
    """
    import hashlib
    
    # Model identifier
    model_name = type(model).__name__
    
    # Data shape
    shape_str = str(getattr(X_data, "shape", None))
    
    # Hash first and last rows to catch data changes
    try:
        X_array = np.asarray(X_data)
        # Sample a few rows for hashing (front, middle, back)
        idx_sample = np.linspace(0, len(X_array) - 1, min(3, len(X_array)), dtype=int)
        sample = X_array[idx_sample]
        data_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
    except Exception:
        # If hashing fails, use random component (no cache)
        import uuid
        data_hash = str(uuid.uuid4())[:8]
    
    return f"{model_name}-{shape_str}-{data_hash}"
```

---

### 6. PDP Feature Name Wrong (xai_service.py:92)

**Problem:** Maps to wrong feature name
```python
# Current (WRONG)
return compute_pdp_1d(
    model, X, int(feature_idxs), 
    feature_name=feature_names[0] if feature_names else None  # Always index 0!
)
```

**Fix:**
```python
if isinstance(feature_idxs, (list, tuple)) and len(feature_idxs) == 2:
    return compute_pdp_2d(model, X, tuple(feature_idxs), feature_names=feature_names)
else:
    idx = int(feature_idxs)
    # Get the correct feature name for the selected index
    if feature_names and idx < len(feature_names):
        fname = feature_names[idx]
    else:
        fname = f"Feature {idx}"
    
    return compute_pdp_1d(model, X, idx, feature_name=fname)
```

---

## IMPORTANT ISSUES — FIX THIS WEEK

### 1. LIME predict_fn Fallback (lime_explainer.py:53-54)

```python
# Current
predict_fn = (
    model.predict_proba
    if hasattr(model, "predict_proba")
    else model.predict
)

# Fixed
predict_fn = None
if hasattr(model, "predict_proba"):
    try:
        # Test that predict_proba actually works
        test_pred = model.predict_proba(X_train[:1])
        predict_fn = model.predict_proba
    except (ValueError, AttributeError, TypeError):
        logger.warning("predict_proba exists but failed on test sample, using predict instead")

if predict_fn is None:
    predict_fn = model.predict
```

---

### 2. Model Compatibility Check (Add to upload_view.py)

```python
def _on_upload(self):
    filepath = filedialog.askopenfilename(...)
    if not filepath:
        return
    
    try:
        model, metadata = self._service.load_external_model(filepath)
        
        # NEW: Check compatibility
        pr = self._service.preprocessing_result
        if pr:
            is_compat, msg = self._service.check_model_compatibility(metadata)
            if not is_compat:
                show_error(_("upload_err_title"), msg)
                return
        
    except Exception as exc:
        show_error(_("upload_err_title"), str(exc))
        logger.error("Failed to load model: %s", exc)
        return

    self._display_model_info(filepath, metadata)
    show_info(_("upload_success_title"), f"Model loaded successfully")
```

Add to `services/pipeline_service.py`:
```python
def check_model_compatibility(self, model_metadata: dict) -> tuple[bool, str]:
    """Check if loaded model is compatible with current preprocessing."""
    pr = self.preprocessing_result
    if pr is None:
        return False, "No preprocessing result. Run preprocessing first."
    
    # Check feature count
    n_features_model = model_metadata.get("n_features")
    if n_features_model and n_features_model != pr.X_test.shape[1]:
        return False, (
            f"Feature mismatch: model expects {n_features_model} features, "
            f"but preprocessing produced {pr.X_test.shape[1]}"
        )
    
    # Check task type
    task_type_model = model_metadata.get("task_type")
    if task_type_model and task_type_model != pr.task_type:
        return False, (
            f"Task type mismatch: model is '{task_type_model}', "
            f"but data is '{pr.task_type}'"
        )
    
    return True, "Compatible"
```

---

### 3. Regression Fairness Check (fairness_view.py:114)

```python
def _run_analysis(self):
    # ... existing code ...
    
    pr = self._service.preprocessing_result
    if pr is None:
        show_error(_("fair_err_title"), _("fair_err_no_data"))
        return
    
    # NEW: Block regression
    if pr.task_type != 'classification':
        show_error(
            _("fair_err_title"),
            f"Fairness analysis only supports classification models. "
            f"Current task type: '{pr.task_type}'"
        )
        return
```

---

### 4. Instance Index Validation (xai_view.py:197)

```python
def _run_shap(self):
    # ... existing code ...
    
    instance_idx = int(self._shap_instance.get())
    
    # NEW: Validate index
    max_idx = len(X_test) - 1
    if instance_idx > max_idx:
        show_error(
            _("xai_err_title"),
            f"Instance index {instance_idx} out of range (0-{max_idx})"
        )
        return
    
    # Rest of method...
```

---

### 5. Sensitive Column Validation (fairness_view.py:103)

```python
def _run_analysis(self):
    sensitive_col = self._sensitive_combo.get()
    if not sensitive_col:
        show_error(_("fair_err_title"), _("fair_err_no_sens"))
        return
    
    # NEW: Validate not a target column
    target_cols = self._service.target_columns or []
    if sensitive_col in target_cols:
        show_error(
            _("fair_err_title"),
            f"Cannot use target column '{sensitive_col}' as sensitive attribute"
        )
        return
    
    # ... rest of method
```

---

## TESTING CHECKLIST

After applying fixes, test:

- [ ] SHAP with 3-class, 10-class classification
- [ ] SHAP with KernelExplainer on 1000+ feature dataset (should warn)
- [ ] Feature importance on linear model
- [ ] LIME on OneClassSVM
- [ ] Fairness with gender/race as sensitive attribute
- [ ] Upload model with different feature count
- [ ] Regression fairness analysis (should block)
- [ ] XAI view with instance_idx > dataset size

---

## IMPLEMENTATION ORDER

1. **First 30 mins:** Fix critical SHAP issues (#1, #2) — these are causing crashes
2. **Next 30 mins:** Fix permutation importance (#3) and PDP names (#6)
3. **Next 1 hour:** Refactor fairness view to use service (#4)
4. **Next 1 hour:** LIME error handling and upload validations (#5)
5. **Next 1 hour:** Instance validation and regression blocking
6. **Final 30 mins:** Test and documentation

---

**Total Estimated Time:** 5-6 hours

**Commit Message:**
```
fix(phase2): Address 6 critical issues

- Fix SHAP multiclass crash (select highest mean absolute class)
- Add warning for slow KernelExplainer
- Safe extraction of permutation importance results
- Move fairness index reconstruction to PipelineService
- Improve SHAP cache key with data hashing
- Fix PDP feature name mapping bug

Also includes:
- Better LIME predict_fn fallback handling
- Model compatibility validation on upload
- Block regression fairness analysis
- Instance index range validation
- Sensitive attribute validation

Fixes: #42, #43, #44, #45, #46, #47
```
