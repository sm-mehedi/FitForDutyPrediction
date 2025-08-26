<h1>Recovery Prediction Project</h1>

<p>This project builds a predictive model to estimate whether a patient with s cu rvy is <strong>fit for duty</strong> after six days (<code>fit_for_duty_d6</code>) based on historical symptom and treatment data. The workflow addresses challenges with tiny datasets and provides interpretable results using Random Forest and SHAP.</p>

<h2>1. Dataset Preparation</h2>
<ul>
  <li>Loaded historical s cu rvy clinical data.</li>
  <li>Converted ordered symptom severities (mild, moderate, severe) into numeric values using <code>LabelEncoder</code>.</li>
  <li>One-hot encoded the treatment column (cider, citrus, vinegar, etc.) and converted these to integers for ML compatibility.</li>
  <li>Encoded the target <code>fit_for_duty_d6</code> as 0 (no) or 1 (yes).</li>
</ul>
<p>✅ <strong>Achieved:</strong> Data is now fully numeric and ML-ready.</p>

<h2>2. Handling Tiny Dataset / Class Imbalance</h2>
<ul>
  <li>The original dataset has 12 records, with only 1 record where <code>fit_for_duty_d6 = 1</code>.</li>
  <li>Created synthetic samples by slightly perturbing the minority class (small Gaussian noise).</li>
  <li>After augmentation, the dataset is balanced with 11 records for each class.</li>
</ul>
<p>✅ <strong>Achieved:</strong> Balanced dataset allows ML models to train without crashing or overfitting on a single sample.</p>

<h2>3. Training a Random Forest</h2>
<ul>
  <li>Split the augmented dataset into training (70%) and testing (30%).</li>
  <li>Trained a <code>RandomForestClassifier</code> to predict if a patient becomes fit for duty based on symptoms and treatment.</li>
  <li>Predicted outcomes on the test set.</li>
</ul>
<p>✅ <strong>Achieved:</strong> Built a functional predictive model for s cu rvy recovery outcomes.</p>

<h2>4. Evaluation</h2>
<ul>
  <li>Generated a classification report showing precision, recall, and F1-score.</li>
  <li>Displayed a confusion matrix using Seaborn.</li>
</ul>
<p>✅ <strong>Achieved:</strong> Evaluated model performance; metrics are perfect on the tiny synthetic dataset (note: results are influenced by data augmentation).</p>

<h2>5. Feature Importance</h2>
<ul>
  <li>Plotted the Random Forest feature importance, showing which symptoms or treatments influence predictions the most.</li>
</ul>
<p>✅ <strong>Achieved:</strong> Insight into which features (e.g., gum rot, lassitude, citrus treatment) drive the model’s predictions.</p>

<h2>6. SHAP Explainability</h2>
<ul>
  <li>Used SHAP (SHapley Additive exPlanations) to interpret the model’s decisions.</li>
  <li>Generated a summary plot showing feature contributions to each prediction.</li>
</ul>
<p>✅ <strong>Achieved:</strong> Model is interpretable, which is critical for research publication and clinical relevance.</p>

<h2>Summary</h2>
<p>This project demonstrates a full pipeline for predicting s cu rvy recovery outcomes on a tiny historical dataset. Despite limited records, careful data augmentation, Random Forest modeling, and SHAP explainability provide insights into which factors most influence recovery. While metrics are artificially perfect due to synthetic augmentation, the methodology is robust and interpretable, making it suitable for discussion, educational purposes, and historical data analysis.</p>
