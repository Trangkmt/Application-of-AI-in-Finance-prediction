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

# Update file organization - consolidated directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PREDICTIONS_FILENAME = 'predictions'

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
    
    res=[]
    for name,pipe in models.items():
        pipe.fit(X_tr,y_tr); pred=pipe.predict(X_te)
        res.append({'Model':name,
                    'Accuracy':accuracy_score(y_te,pred),
                    'F1_macro':f1_score(y_te,pred,average='macro')})

        ConfusionMatrixDisplay.from_predictions(y_te,pred,labels=[0,1,2])
        plt.title(f"CM – {name}")
        cm_path = DATA_DIR / f'cm_{session_id}_{name}.png'
        plt.savefig(cm_path, dpi=300)
        plt.clf()

        # Save report with consistent naming
        report_filename = f'class_report_{session_id}_{name}.txt'
        report_path = DATA_DIR / report_filename
        with open(report_path, 'w') as f:
            f.write(classification_report(y_te,pred,digits=3))

        model_path = DATA_DIR / f'model_{session_id}_{name}.pkl'
        joblib.dump(pipe, model_path)

    results_path = DATA_DIR / f'results_{session_id}.csv'
    pd.DataFrame(res).to_csv(results_path, index=False)

    rf=models['RandomForest']
    df['Rating_pred']=rf.predict(X)
    df['Prob_Good']=rf.predict_proba(X)[:,-1].round(3)
    predictions_path = DATA_DIR / f'predictions_{session_id}.csv'
    df.to_csv(predictions_path, index=False)
    
    rf_model_path = DATA_DIR / f'model_{session_id}_{MODEL_FILE}'
    joblib.dump(rf, rf_model_path)
    
    return True, {
        'session_id': session_id,
        'results_path': str(results_path),
        'model_path': str(rf_model_path),
        'cm_paths': [str(DATA_DIR / f'cm_{session_id}_{name}.png') for name in models.keys()],
        'report_paths': [f'/view_report/{session_id}/{name}' for name in models.keys()],
        'predictions_path': str(predictions_path)
    }

# ==================== FLASK WEB APP ====================
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'), 
           static_folder=str(BASE_DIR / 'static'))
app.secret_key = 'finance-analysis-secret-key'
app.config['UPLOAD_FOLDER'] = str(DATA_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

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
        
        # Updated model loading
        latest_model = None
        max_time = 0
        for model_file in DATA_DIR.glob(f'model_*_{MODEL_FILE}'):
            if model_file.stat().st_mtime > max_time:
                max_time = model_file.stat().st_mtime
                latest_model = model_file
        
        if not latest_model:
            flash('No model found. Please train a model first.')
            return redirect(url_for('index'))
        
        model = joblib.load(latest_model)
        
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
        # Save uploaded file directly to DATA_DIR
        filename = str(uuid.uuid4()) + Path(file.filename).suffix
        file_path = DATA_DIR / filename
        file.save(file_path)
        
        # Process file
        success, result = run_pipeline(file_path)
        
        if not success:
            flash(result)
            return redirect(url_for('index'))
        
        # Read the results CSV to get models data
        session_id = result['session_id']
        results_df = pd.read_csv(result['results_path'])
        models_data = results_df.to_dict('records')
        
        # Fix: Update paths to use correct format for routing
        cm_paths = [f'/file/{session_id}/cm_{model}' for model in results_df['Model']]
        
        # Create enhanced results dictionary
        enhanced_results = {
            'session_id': session_id,
            'models_data': models_data,
            'cm_paths': cm_paths,
            'report_paths': result['report_paths'],
            'predictions_path': f'/view_csv/{session_id}'
        }
        
        # Display results
        return render_template('training_results.html', results=enhanced_results)

@app.route('/file/<path:session_id>/<path:filename>')
def serve_file(session_id, filename):
    """Serve files from data directory"""
    # Fix: Properly construct filename with session ID
    if filename.startswith('cm_'):
        # Handle confusion matrix files (cm_Logistic, cm_RandomForest)
        model_name = filename.replace('cm_', '')
        full_filename = f"cm_{session_id}_{model_name}.png"
    elif '.' in filename:
        # Handle files with explicit extensions
        base, ext = filename.rsplit('.', 1)
        full_filename = f"{base}_{session_id}.{ext}"
    else:
        # Handle other files (assume CSV if no extension)
        full_filename = f"{filename}_{session_id}.csv"
    
    # Debug output to help troubleshoot
    print(f"Serving file: {full_filename} from {DATA_DIR}")
    
    if not (DATA_DIR / full_filename).exists():
        print(f"File not found: {DATA_DIR / full_filename}")
        return f"File {full_filename} not found", 404
        
    return send_from_directory(DATA_DIR, full_filename, as_attachment=False)

@app.route('/download_report/<path:session_id>/<path:model_name>')
def download_report(session_id, model_name):
    """Download classification report as text file"""
    report_filename = f'class_report_{session_id}_{model_name}.txt'
    
    if not (DATA_DIR / report_filename).exists():
        flash("Classification report not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, report_filename, as_attachment=True)

@app.route('/download_csv/<path:session_id>/<path:filename>')
def download_csv(session_id, filename):
    """Download CSV file"""
    file_path = DATA_DIR / f'{filename}_{session_id}.csv'
    
    if not file_path.exists():
        flash("File not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, file_path.name, as_attachment=True)

@app.route('/download_model/<path:session_id>')
def download_model(session_id):
    """Download the trained RandomForest model"""
    model_filename = f'model_{session_id}_{MODEL_FILE}'
    
    if not (DATA_DIR / model_filename).exists():
        flash("Model file not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, model_filename, as_attachment=True)

@app.route('/view_report/<path:session_id>/<path:model_name>')
def view_report(session_id, model_name):
    """View classification report directly in browser"""
    try:
        report_filename = f'class_report_{session_id}_{model_name}.txt'
        file_path = DATA_DIR / report_filename
        
        if not file_path.exists():
            print(f"Report file not found: {file_path}")
            flash("Không tìm thấy báo cáo phân loại")
            return redirect(url_for('index'))
        
        try:
            with open(file_path, 'r') as f:
                report_content = f.read()
            
            # Đọc các thống kê từ báo cáo
            import re
            accuracy_match = re.search(r'accuracy\s*:\s*(0\.\d+)', report_content)
            precision_match = re.search(r'macro avg\s+(\d+\.\d+)', report_content)
            recall_match = re.search(r'macro avg\s+\d+\.\d+\s+(\d+\.\d+)', report_content)
            f1_match = re.search(r'macro avg\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)', report_content)
            
            # Định dạng các giá trị để hiển thị
            from datetime import datetime
            
            accuracy = accuracy_match.group(1) if accuracy_match else "N/A"
            precision = precision_match.group(1) if precision_match else "N/A"
            recall = recall_match.group(1) if recall_match else "N/A"
            f1 = f1_match.group(1) if f1_match else "N/A"
            
            # Render template với dữ liệu
            return render_template('view_report.html',
                                  report=report_content,
                                  model=model_name,
                                  session_id=session_id,
                                  accuracy=accuracy,
                                  precision=precision,
                                  recall=recall,
                                  f1=f1,
                                  now=datetime.now())
            
        except Exception as e:
            print(f"Error reading report file: {str(e)}")
            flash(f"Lỗi khi đọc file báo cáo: {str(e)}")
            return redirect(url_for('index'))
            
    except Exception as e:
        import traceback
        print(f"Error in view_report: {str(e)}")
        print(traceback.format_exc())
        flash(f"Lỗi khi xem báo cáo: {str(e)}")
        return redirect(url_for('index'))

@app.route('/view_csv/<session_id>/<filename>')
@app.route('/view_csv/<session_id>', defaults={'filename': DEFAULT_PREDICTIONS_FILENAME})
def view_csv(session_id, filename):
    try:
        # Fix: Construct the path to the CSV file using standardized naming
        csv_path = DATA_DIR / f'{filename}_{session_id}.csv'
        
        print(f"Looking for CSV file: {csv_path}")
        
        # Check if file exists
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}")
            # Try alternate filename if using legacy format
            if filename == 'predictions_with_prob.csv':
                alt_path = DATA_DIR / f'predictions_{session_id}.csv'
                if alt_path.exists():
                    csv_path = alt_path
                    print(f"Found alternative CSV file: {alt_path}")
                else:
                    flash(f"File not found: {filename}")
                    return render_template('view_csv.html',
                                      data=None,
                                      filename=filename,
                                      session_id=session_id)
        
        # Load the CSV file
        try:
            data = pd.read_csv(csv_path)
            return render_template('view_csv.html',
                                  data=data,
                                  filename=filename,
                                  session_id=session_id)
        except Exception as e:
            print(f"CSV parsing error: {str(e)}")
            flash(f"Error parsing CSV file: {str(e)}")
            return render_template('view_csv.html',
                                  data=None,
                                  filename=filename,
                                  session_id=session_id)
    except Exception as e:
        print(f"Error in view_csv route: {str(e)}")
        flash(f"Error loading CSV file: {str(e)}")
        return render_template('view_csv.html',
                              data=None,
                              filename=filename,
                              session_id=session_id)

@app.route('/training_results/<session_id>')
def training_results(session_id):
    """Redirect to view_prediction_results for backward compatibility"""
    return redirect(url_for('view_prediction_results', session_id=session_id))

@app.route('/view_prediction_results/<path:session_id>')
def view_prediction_results(session_id):
    """View complete analysis and visualization dashboard"""
    try:
        # Read results file using new naming convention
        results_path = DATA_DIR / f'results_{session_id}.csv'
        predictions_path = DATA_DIR / f'predictions_{session_id}.csv'
        
        if not results_path.exists() or not predictions_path.exists():
            flash("Không tìm thấy tệp kết quả")
            return redirect(url_for('index'))
        
        # Read data
        results_df = pd.read_csv(results_path)
        predictions_df = pd.read_csv(predictions_path)
        
        # Prepare data for template
        models_data = results_df.to_dict('records')
        cm_paths = [f'/file/{session_id}/cm_{model}' for model in results_df['Model']]
        report_paths = [f'/view_report/{session_id}/{model}' for model in results_df['Model']]
        
        # Get statistics from predictions
        stats = {
            'total_samples': len(predictions_df),
            'good_count': (predictions_df['Rating_pred'] == 2).sum(),
            'avg_count': (predictions_df['Rating_pred'] == 1).sum(),
            'poor_count': (predictions_df['Rating_pred'] == 0).sum(),
            'avg_prob_good': predictions_df['Prob_Good'].mean()
        }
        
        # Render template instead of generating HTML directly
        return render_template('view_prediction_results.html',
                               session_id=session_id,
                               models_data=models_data,
                               cm_paths=cm_paths,
                               report_paths=report_paths,
                               stats=stats)
        
    except Exception as e:
        import traceback
        print(f"Lỗi khi xem kết quả dự đoán: {str(e)}")
        print(traceback.format_exc())
        flash(f"Lỗi khi xem kết quả dự đoán: {str(e)}")
        return redirect(url_for('index'))

# ==================== ENTRY ====================
if __name__=="__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        run_pipeline(sys.argv[1])      # huấn luyện qua Terminal
    else:
        # Run web app
        app.run(debug=True, host='0.0.0.0', port=5000)