# finance_app.py  –  24-04-2025
# ================================================================
# • Auto-detect CSV encoding + delimiter   (chardet + csv.Sniffer)
# • Tính thêm AssetTurn, EPS_Growth (nếu đủ cột)
# • Gán nhãn cứng (ngưỡng nới lỏng) ⇒ 0:Poor 1:Avg 2:Good
# • Train Logistic + RandomForest + NaiveBayes, xuất:
#     results_classification.csv
#     predictions_with_prob.csv
#     cm_*.png, class_report_*.txt
#     model_Logistic.pkl, model_RandomForest.pkl, model_NaiveBayes.pkl
# • Web Interface: nhập 6 chỉ số hoặc chọn CSV/XLSX và train/dự đoán hàng loạt
# =================================================================
import sys, csv, warnings, joblib, chardet, os
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
import uuid
from datetime import datetime
from functools import wraps

warnings.filterwarnings("ignore")

FEATURES   = ['ROE','ROA','DebtEq','CurrRatio','AssetTurn','EPS_Growth']
MODEL_FILE = 'model_RandomForest.pkl'

# Update file organization - consolidated directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PREDICTIONS_FILENAME = 'predictions'

# ---------- Helper function for standardized file paths ----------
def get_session_path(session_id, file_type, name=None):
    """Generate standardized file paths for session artifacts"""
    if file_type == 'eda_dir':
        return DATA_DIR / f'eda_{session_id}'
    elif file_type == 'model':
        model_name = name or MODEL_FILE
        return DATA_DIR / f'model_{session_id}_{model_name}.pkl'
    elif file_type == 'cm':
        model_name = name or ''
        return DATA_DIR / f'cm_{session_id}_{model_name}.png'
    elif file_type == 'report':
        model_name = name or ''
        return DATA_DIR / f'class_report_{session_id}_{model_name}.txt'
    elif file_type == 'results':
        return DATA_DIR / f'results_{session_id}.csv'
    elif file_type == 'predictions':
        return DATA_DIR / f'predictions_{session_id}.csv'
    else:
        return DATA_DIR / f'{file_type}_{session_id}.csv'

# ---------- đọc CSV bất kỳ ----------
def read_any_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()[:40000]
    enc = chardet.detect(raw)['encoding'] or 'latin1'
    sample = raw.decode(enc, errors='ignore')
    sep = csv.Sniffer().sniff(sample).delimiter
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

# ---------- tạo các biểu đồ EDA ----------
def eda_visuals(df: pd.DataFrame, session_id):
    """Tạo các biểu đồ phân tích khám phá dữ liệu và lưu vào thư mục data"""
    eda_dir = get_session_path(session_id, 'eda_dir')
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    cols_to_plot = [col for col in FEATURES if col in df.columns]

    # 1. Correlation Heatmap
    corr = df[cols_to_plot].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Ma trận tương quan giữa các biến")
    plt.savefig(f"{eda_dir}/heatmap_correlation_{session_id}.png", dpi=300)
    plt.close()

    # 2. Boxplot theo Rating (nếu có)
    if 'Rating' in df.columns:
        n = len(cols_to_plot)
        ncols = 2
        nrows = (n + 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cols_to_plot):
            sns.boxplot(x='Rating', y=col, data=df, ax=axes[i])
            axes[i].set_title(f"{col} theo Rating")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{eda_dir}/boxplot_by_rating_{session_id}.png", dpi=300)
        plt.close()

    # 3. Pairplot (nếu có Rating)
    if 'Rating' in df.columns and len(df) <= 1000:  # Giới hạn kích thước để tránh biểu đồ quá lớn
        sns.pairplot(df[cols_to_plot + ['Rating']], hue='Rating', palette='Set2')
        plt.savefig(f"{eda_dir}/pairplot_by_rating_{session_id}.png", dpi=300)
        plt.close()
    
    return eda_dir

# Helper to handle common route errors
def handle_route_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Lỗi: {str(e)}", "danger")
            return redirect(url_for('index'))
    return wrapper

# Helper to load analysis data
def load_analysis_data(session_id):
    """Load common analysis data used by multiple routes"""
    results_path = get_session_path(session_id, 'results')
    predictions_path = get_session_path(session_id, 'predictions')
    
    if not results_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Không tìm thấy dữ liệu phân tích cho phiên này")
    
    results_df = pd.read_csv(results_path)
    predictions_df = pd.read_csv(predictions_path)
    
    models_data = results_df.to_dict('records')
    cm_paths = [f'/file/{session_id}/cm_{model}' for model in results_df['Model']]
    report_paths = [f'/view_report/{session_id}/{model}' for model in results_df['Model']]
    
    stats = {
        'total_samples': len(predictions_df),
        'good_count': (predictions_df['Rating_pred'] == 2).sum(),
        'avg_count': (predictions_df['Rating_pred'] == 1).sum(),
        'poor_count': (predictions_df['Rating_pred'] == 0).sum(),
        'avg_prob_good': predictions_df['Prob_Good'].mean()
    }
    
    return {
        'models_data': models_data,
        'cm_paths': cm_paths,
        'report_paths': report_paths,
        'stats': stats,
        'predictions_df': predictions_df
    }

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
    
    session_id = str(uuid.uuid4())[:8]
    
    # Tạo các biểu đồ phân tích dữ liệu
    eda_path = eda_visuals(df, session_id)

    models={
        'Logistic':Pipeline([('sc',StandardScaler()),
                             ('lg',LogisticRegression(max_iter=1000,multi_class='multinomial'))]),
        'RandomForest':Pipeline([('sc',StandardScaler()),
                                 ('rf',RandomForestClassifier(n_estimators=400,
                                                              class_weight='balanced',
                                                              random_state=42))]),
        'NaiveBayes':Pipeline([('sc',StandardScaler()),
                               ('nb',GaussianNB())])
    }
    
    res=[]
    for name,pipe in models.items():
        pipe.fit(X_tr,y_tr); pred=pipe.predict(X_te)
        res.append({'Model':name,
                    'Accuracy':accuracy_score(y_te,pred),
                    'F1_macro':f1_score(y_te,pred,average='macro')})

        ConfusionMatrixDisplay.from_predictions(y_te,pred,labels=[0,1,2])
        plt.title(f"CM – {name}")
        cm_path = get_session_path(session_id, 'cm', name)
        plt.savefig(cm_path, dpi=300)
        plt.clf()

        # Save report with consistent naming
        report_path = get_session_path(session_id, 'report', name)
        with open(report_path, 'w') as f:
            f.write(classification_report(y_te,pred,digits=3))

        model_path = get_session_path(session_id, 'model', name)
        joblib.dump(pipe, model_path)

    results_path = get_session_path(session_id, 'results')
    pd.DataFrame(res).to_csv(results_path, index=False)

    rf=models['RandomForest']
    df['Rating_pred']=rf.predict(X)
    df['Prob_Good']=rf.predict_proba(X)[:,-1].round(3)
    predictions_path = get_session_path(session_id, 'predictions')
    df.to_csv(predictions_path, index=False)
    
    rf_model_path = get_session_path(session_id, 'model')
    joblib.dump(rf, rf_model_path)
    
    return True, {
        'session_id': session_id,
        'results_path': str(results_path),
        'model_path': str(rf_model_path),
        'cm_paths': [str(get_session_path(session_id, 'cm', name)) for name in models.keys()],
        'report_paths': [f'/view_report/{session_id}/{name}' for name in models.keys()],
        'predictions_path': str(predictions_path),
        'eda_paths': [f'/file/{session_id}/eda/{Path(p).stem}' for p in eda_path.glob('*.png')]
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

@app.route('/dashboard')
@app.route('/dashboard/<path:session_id>')
def dashboard(session_id=None):
    """Show dashboard when accessing /dashboard directly"""
    # If session_id is provided in URL, use it
    if session_id:
        session['analysis_session_id'] = session_id
        return redirect(url_for('analysis_dashboard', session_id=session_id))
    
    # Otherwise check if there's a session_id in the session state
    session_id = session.get('analysis_session_id')
    if session_id:
        return redirect(url_for('analysis_dashboard', session_id=session_id))
    else:
        flash("No active analysis session found. Please upload a file to analyze.", "info")
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
            flash(result, "danger")
            return redirect(url_for('index'))
        
        # Lưu session_id vào session state để phục vụ điều hướng sau này
        session_id = result['session_id']
        session['analysis_session_id'] = session_id  # Store in session
        session['current_analysis'] = {
            'session_id': session_id,
            'filename': file.filename,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': FEATURES
        }
        
        # Chuyển hướng đến trang analysis_dashboard với kết quả phân tích trực quan
        return redirect(url_for('analysis_dashboard', session_id=session_id))

@app.route('/analysis_dashboard/<path:session_id>')
@handle_route_errors
def analysis_dashboard(session_id):
    """Hiển thị dashboard tích hợp với tất cả kết quả phân tích"""
    # Store the session ID in the server-side session
    session['analysis_session_id'] = session_id
    
    # Load common analysis data
    data = load_analysis_data(session_id)
    
    # Kiểm tra xem có biểu đồ EDA không
    eda_dir = get_session_path(session_id, 'eda_dir')
    has_eda = eda_dir.exists() and any(eda_dir.glob('*.png'))
    
    # Lấy danh sách các biểu đồ EDA nếu có
    eda_paths = []
    if has_eda:
        eda_files = list(eda_dir.glob('*.png'))
        eda_paths = [f'/file/{session_id}/eda/{Path(p).name}' for p in eda_files]
    
    # Trả về dashboard tích hợp
    return render_template('analysis_dashboard.html',
                          session_id=session_id,
                          models_data=data['models_data'],
                          cm_paths=data['cm_paths'],
                          report_paths=data['report_paths'],
                          stats=data['stats'],
                          has_eda=has_eda,
                          eda_paths=eda_paths,
                          analysis_info=session.get('current_analysis', {}),
                          preview_data=data['predictions_df'].head(10))

@app.route('/file/<path:session_id>/<path:filename>')
def serve_file(session_id, filename):
    """Serve files from data directory"""
    # Xử lý các tệp confusion matrix
    if filename.startswith('cm_'):
        model_name = filename.replace('cm_', '')
        full_filename = f"cm_{session_id}_{model_name}.png"
        return send_from_directory(DATA_DIR, full_filename, as_attachment=False)
    
    # Xử lý các tệp EDA
    elif filename.startswith('eda/'):
        eda_name = filename.replace('eda/', '')
        eda_dir = get_session_path(session_id, 'eda_dir')
        
        # Tìm tệp EDA phù hợp
        for eda_file in eda_dir.glob('*.png'):
            if eda_name in eda_file.name:
                return send_from_directory(eda_dir, eda_file.name, as_attachment=False)
        
        return f"EDA file not found: {eda_name}", 404
    
    # Xử lý các tệp CSV và tệp khác
    elif '.' in filename:
        base, ext = filename.rsplit('.', 1)
        full_filename = f"{base}_{session_id}.{ext}"
    else:
        full_filename = f"{filename}_{session_id}.csv"
    
    if not (DATA_DIR / full_filename).exists():
        return f"File {full_filename} not found", 404
        
    return send_from_directory(DATA_DIR, full_filename, as_attachment=False)

@app.route('/download_report/<path:session_id>/<path:model_name>')
def download_report(session_id, model_name):
    report_filename = f'class_report_{session_id}_{model_name}.txt'
    
    if not (DATA_DIR / report_filename).exists():
        flash("Classification report not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, report_filename, as_attachment=True)

@app.route('/download_csv/<path:session_id>/<path:filename>')
def download_csv(session_id, filename):
    file_path = get_session_path(session_id, filename)
    
    if not file_path.exists():
        flash("File not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, file_path.name, as_attachment=True)

@app.route('/download_model/<path:session_id>')
@app.route('/download_model/<path:session_id>/<path:model_name>')
def download_model(session_id, model_name=None):
    model_path = get_session_path(session_id, 'model', model_name)
    
    if not model_path.exists():
        flash("Model file not found")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, model_path.name, as_attachment=True)

@app.route('/view_report/<path:session_id>/<path:model_name>')
@handle_route_errors
def view_report(session_id, model_name):
    # Store the session ID in the server-side session
    session['analysis_session_id'] = session_id
    
    report_path = get_session_path(session_id, 'report', model_name)
    
    if not report_path.exists():
        flash("Không tìm thấy báo cáo phân loại")
        return redirect(url_for('index'))
    
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    import re
    accuracy_match = re.search(r'accuracy\s*:\s*(0\.\d+)', report_content)
    precision_match = re.search(r'macro avg\s+(\d+\.\d+)', report_content)
    recall_match = re.search(r'macro avg\s+\d+\.\d+\s+(\d+\.\d+)', report_content)
    f1_match = re.search(r'macro avg\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)', report_content)
    
    accuracy = accuracy_match.group(1) if accuracy_match else "N/A"
    precision = precision_match.group(1) if precision_match else "N/A"
    recall = recall_match.group(1) if recall_match else "N/A"
    f1 = f1_match.group(1) if f1_match else "N/A"
    
    return render_template('view_report.html',
                          report=report_content,
                          model=model_name,
                          session_id=session_id,
                          accuracy=accuracy,
                          precision=precision,
                          recall=recall,
                          f1=f1,
                          now=datetime.now())

@app.route('/view_csv/<session_id>/<filename>')
@app.route('/view_csv/<session_id>', defaults={'filename': DEFAULT_PREDICTIONS_FILENAME})
@handle_route_errors
def view_csv(session_id, filename):
    # Store the session ID in the server-side session
    session['analysis_session_id'] = session_id
    
    csv_path = get_session_path(session_id, filename)
    
    if not csv_path.exists():
        if filename == 'predictions_with_prob.csv':
            alt_path = get_session_path(session_id, 'predictions')
            if alt_path.exists():
                csv_path = alt_path
            else:
                flash(f"File not found: {filename}")
                return render_template('view_csv.html',
                                  data=None,
                                  filename=filename,
                                  session_id=session_id)
    
    data = pd.read_csv(csv_path)
    return render_template('view_csv.html',
                          data=data,
                          filename=filename,
                          session_id=session_id)

@app.route('/training_results/<session_id>')
def training_results(session_id):
    return redirect(url_for('view_prediction_results', session_id=session_id))

@app.route('/view_prediction_results/<path:session_id>')
@handle_route_errors
def view_prediction_results(session_id):
    # Store the session ID in the server-side session
    session['analysis_session_id'] = session_id
    
    # Use helper function to load analysis data
    data = load_analysis_data(session_id)
    
    # Return the template with data
    return render_template('view_prediction_results.html',
                          session_id=session_id,
                          models_data=data['models_data'],
                          cm_paths=data['cm_paths'],
                          report_paths=data['report_paths'],
                          stats=data['stats'])

# ==================== ENTRY ====================
if __name__=="__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        run_pipeline(sys.argv[1])      # huấn luyện qua Terminal
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)