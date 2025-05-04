# finance_app.py  –  24-04-2025
# ================================================================
# • Auto-detect CSV encoding + delimiter   (chardet + csv.Sniffer)
# • Tính thêm AssetTurn, EPS_Growth (nếu đủ cột)
# • Gán nhãn cứng (ngưỡng nới lỏng) ⇒ 0:Poor 1:Avg 2:Good
# • Train Logistic + RandomForest, xuất:
#     results_classification.csv
#     predictions_with_prob.csv
#     cm_*.png, class_report_*.txt
#     model_Logistic.pkl, model_RandomForest.pkl
# • Web Interface: nhập 6 chỉ số hoặc chọn CSV/XLSX và train/dự đoán hàng loạt
# =================================================================
import sys, csv, warnings, joblib, chardet, os, io
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import uuid

warnings.filterwarnings("ignore")

FEATURES   = ['ROE','ROA','DebtEq','CurrRatio','AssetTurn','EPS_Growth']
MODEL_FILE = 'model_RandomForest.pkl'
OUTPUT_DIR = Path('static')
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR = OUTPUT_DIR / 'class_report'
REPORT_DIR.mkdir(exist_ok=True)

# ---------- đọc CSV bất kỳ ----------
def read_any_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()[:40000]
    enc = chardet.detect(raw)['encoding'] or 'latin1'
    sample = raw.decode(enc, errors='ignore')
    sep = csv.Sniffer().sniff(sample).delimiter
    print(f"📥  Loading CSV  |  enc={enc}  |  sep='{sep}'")
    return pd.read_csv(path, encoding=enc, sep=sep)

# ---------- đổi tên cột ----------
def smart_rename(df: pd.DataFrame) -> pd.DataFrame:
    ren={}
    for c in df.columns:
        low=c.lower()
        if 'returnonequity' in low: ren[c]='ROE'
        if 'returnonassets'  in low: ren[c]='ROA'
        if 'debttoequity'    in low: ren[c]='DebtEq'
        if 'currentratio'    in low: ren[c]='CurrRatio'
        if 'assetturnover'   in low: ren[c]='AssetTurn'
        if 'earningspershare' in low or low=='eps': ren[c]='EPS'
        if 'totalrevenue'    in low: ren[c]='Revenue'
        if 'totalassets'     in low: ren[c]='Assets'
    return df.rename(columns=ren)

# ---------- hàm gán nhãn ----------
def build_labeler(has_AT, has_EG):
    def lab(r):
        # ---------- GOOD ----------
        good = (
            (r.ROE       >= 0.04)   &   # 4 %
            (r.ROA       >= 0.015)  &   # 1.5 %
            (r.DebtEq    <= 50)     &   # nợ ≤ 50 lần vốn
            (r.CurrRatio >= 0.6)
        )
        # ---------- POOR ----------
        poor = (
            (r.ROE       < 0.015) |
            (r.ROA       < 0.007) |
            (r.DebtEq    > 50)   |      # rất nhiều nợ
            (r.CurrRatio < 0.4)
        )
        if has_AT:   # chưa có AssetTurn, nhưng giữ cho tương lai
            good &= (r.AssetTurn >= 0.10)
            poor |= (r.AssetTurn < 0.03)
        if has_EG:   # chưa có EPS_Growth
            good &= (r.EPS_Growth >= 0.01)
            poor |= (r.EPS_Growth < -0.07)
        return 2 if good else 0 if poor else 1
    return lab

# ==================== TRAIN PIPELINE ====================
def run_pipeline(file_path):
    p=Path(file_path)
    if not p.exists(): 
        return False, f"❌ File không tồn tại: {p}"

    df = read_any_csv(p) if p.suffix.lower()=='.csv' else pd.read_excel(p)
    df = smart_rename(df)

    # tính thêm cột
    if 'AssetTurn' not in df.columns and {'Revenue','Assets'}.issubset(df.columns):
        df['AssetTurn']=df['Revenue']/df['Assets']
    if 'EPS' in df.columns:
        df['EPS_Growth']=df.groupby('shortName')['EPS'].pct_change().fillna(0)
    else:
        df['EPS_Growth']=0.0

    feats=[f for f in FEATURES if f in df.columns]
    df=df.dropna(subset=feats)

    if 'Rating' not in df.columns:
        df['Rating']=df.apply(build_labeler('AssetTurn'in feats,'EPS_Growth'in feats),axis=1)

    X,y=df[feats],df['Rating']
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

    models={
        'Logistic':Pipeline([('sc',StandardScaler()),
                             ('lg',LogisticRegression(max_iter=1000,multi_class='multinomial'))]),
        'RandomForest':Pipeline([('sc',StandardScaler()),
                                 ('rf',RandomForestClassifier(n_estimators=400,
                                                              class_weight='balanced',
                                                              random_state=42))])
    }
    
    session_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / session_id
    output_path.mkdir(exist_ok=True)
    
    res=[]
    for name,pipe in models.items():
        pipe.fit(X_tr,y_tr); pred=pipe.predict(X_te)
        res.append({'Model':name,
                    'Accuracy':accuracy_score(y_te,pred),
                    'F1_macro':f1_score(y_te,pred,average='macro')})

        ConfusionMatrixDisplay.from_predictions(y_te,pred,labels=[0,1,2])
        plt.title(f"CM – {name}")
        cm_path = output_path / f'cm_{name}.png'
        plt.savefig(cm_path, dpi=300)
        plt.clf()

        # Save report to the dedicated reports directory
        report_filename = f'class_report_{session_id}_{name}.txt'
        report_path = REPORT_DIR / report_filename
        with open(report_path, 'w') as f:
            f.write(classification_report(y_te,pred,digits=3))

        model_path = output_path / f'model_{name}.pkl'
        joblib.dump(pipe, model_path)

    results_path = output_path / 'results_classification.csv'
    pd.DataFrame(res).to_csv(results_path, index=False)

    rf=models['RandomForest']
    df['Rating_pred']=rf.predict(X)
    df['Prob_Good']=rf.predict_proba(X)[:,-1].round(3)
    predictions_path = output_path / 'predictions_with_prob.csv'
    df.to_csv(predictions_path, index=False)
    
    joblib.dump(rf, output_path / MODEL_FILE)
    
    return True, {
        'session_id': session_id,
        'results_path': str(results_path),
        'model_path': str(output_path / MODEL_FILE),
        'cm_paths': [str(output_path / f'cm_{name}.png') for name in models.keys()],
        'report_paths': [f'/view_report/{session_id}/{name}' for name in models.keys()],
        'predictions_path': str(predictions_path)
    }

# ==================== FLASK WEB APP ====================
app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))
app.secret_key = 'finance-analysis-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_one', methods=['POST'])
def predict_one():
    try:
        # Get form values
        values = {}
        for feature in FEATURES:
            value = request.form.get(feature, '')
            if not value:
                flash(f'Please enter a value for {feature}')
                return redirect(url_for('index'))
            values[feature] = float(value)
        
        # Load model
        model_path = Path(MODEL_FILE)
        if not model_path.exists():
            flash('No model found. Please train a model first.')
            return redirect(url_for('index'))
        
        model = joblib.load(model_path)
        
        # Make prediction
        X_new = pd.DataFrame([list(values.values())], columns=FEATURES)
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0][-1]
        
        result = {
            'rating': ['Kém', 'TB', 'Tốt'][prediction],
            'probability': f'{probability:.2%}'
        }
        
        return render_template('result.html', result=result)
    
    except ValueError as e:
        flash(f'Error with input values: {str(e)}')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file:
        # Save uploaded file
        filename = str(uuid.uuid4()) + Path(file.filename).suffix
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(file_path)
        
        # Process file
        success, result = run_pipeline(file_path)
        
        if not success:
            flash(result)
            return redirect(url_for('index'))
        
        # Display results
        return render_template('training_results.html', results=result)

@app.route('/static/<path:session_id>/<path:filename>')
def download_file(session_id, filename):
    directory = OUTPUT_DIR / session_id
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/download_report/<path:session_id>/<path:model_name>')
def download_report(session_id, model_name):
    """Download classification report as text file"""
    report_filename = f'class_report_{session_id}_{model_name}.txt'
    file_path = REPORT_DIR / report_filename
    
    if not file_path.exists():
        flash("Classification report not found")
        return redirect(url_for('index'))
    
    return send_from_directory(REPORT_DIR, report_filename, as_attachment=True)

@app.route('/download_csv/<path:session_id>/<path:filename>')
def download_csv(session_id, filename):
    """Download CSV file"""
    directory = OUTPUT_DIR / session_id
    file_path = directory / filename
    
    if not file_path.exists():
        flash("File not found")
        return redirect(url_for('index'))
    
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/download_model/<path:session_id>')
def download_model(session_id):
    """Download the trained RandomForest model"""
    directory = OUTPUT_DIR / session_id
    model_path = directory / MODEL_FILE
    
    if not model_path.exists():
        flash("Model file not found")
        return redirect(url_for('index'))
    
    return send_from_directory(directory, MODEL_FILE, as_attachment=True)

@app.route('/view_report/<path:session_id>/<path:model_name>')
def view_report(session_id, model_name):
    """View classification report directly in browser"""
    report_filename = f'class_report_{session_id}_{model_name}.txt'
    file_path = REPORT_DIR / report_filename
    
    if not file_path.exists():
        flash("Classification report not found")
        return redirect(url_for('index'))
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    return render_template('view_report.html', 
                           content=content,
                           filename=report_filename,
                           session_id=session_id,
                           model_name=model_name)

@app.route('/view_csv/<path:session_id>/<path:filename>')
def view_csv(session_id, filename):
    """View CSV directly in browser as a table"""
    directory = OUTPUT_DIR / session_id
    file_path = directory / filename
    
    if not file_path.exists():
        flash("File not found")
        return redirect(url_for('index'))
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert to HTML table
    table_html = df.to_html(classes='table table-striped table-bordered', index=False)
    
    return render_template('view_csv.html', 
                          table_html=table_html, 
                          filename=filename,
                          session_id=session_id)

@app.route('/view_prediction_results/<path:session_id>')
def view_prediction_results(session_id):
    """View complete analysis and visualization dashboard"""
    directory = OUTPUT_DIR / session_id
    
    # Read results file
    results_path = directory / 'results_classification.csv'
    predictions_path = directory / 'predictions_with_prob.csv'
    
    if not results_path.exists() or not predictions_path.exists():
        flash("Results files not found")
        return redirect(url_for('index'))
    
    # Read data
    results_df = pd.read_csv(results_path)
    predictions_df = pd.read_csv(predictions_path)
    
    # Prepare data for template
    models_data = results_df.to_dict('records')
    cm_paths = [f'/static/{session_id}/cm_{model}.png' for model in results_df['Model']]
    report_paths = [f'/view_report/{session_id}/{model}' for model in results_df['Model']]
    
    # Get some statistics from predictions
    stats = {
        'total_samples': len(predictions_df),
        'good_count': (predictions_df['Rating_pred'] == 2).sum(),
        'avg_count': (predictions_df['Rating_pred'] == 1).sum(),
        'poor_count': (predictions_df['Rating_pred'] == 0).sum(),
        'avg_prob_good': predictions_df['Prob_Good'].mean()
    }
    
    return render_template('dashboard.html',
                          session_id=session_id,
                          models_data=models_data,
                          cm_paths=cm_paths,
                          report_paths=report_paths,
                          stats=stats,
                          predictions_url=f'/view_csv/{session_id}/predictions_with_prob.csv',
                          zip=zip)  # Add zip function to the template context

# ==================== ENTRY ====================
if __name__=="__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        run_pipeline(sys.argv[1])      # huấn luyện qua Terminal
    else:
        # Run web app
        app.run(debug=True, host='0.0.0.0', port=5000)