# finance_app.py  –  24-04-2025
# ================================================================
# • Tự động phát hiện mã hóa + dấu phân cách CSV   (chardet + csv.Sniffer)
# • Tính thêm AssetTurn,
# • Gán nhãn cứng (ngưỡng nới lỏng) ⇒ 0:Kém 1:Trung bình 2:Tốt
# • Huấn luyện Logistic + RandomForest + NaiveBayes, xuất:
#     results_classification.csv
#     predictions_with_prob.csv
#     cm_*.png, class_report_*.txt
#     model_Logistic.pkl, model_RandomForest.pkl, model_NaiveBayes.pkl
# • Giao diện Web: nhập 6 chỉ số hoặc chọn CSV/XLSX và huấn luyện/dự đoán hàng loạt
# =================================================================
import sys, csv, warnings, joblib, chardet, os
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import seaborn, but fall back to a simple implementation if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Warning: Seaborn could not be imported. Using basic matplotlib visualizations instead.")
    HAS_SEABORN = False
    
    # Define minimal replacement for the required seaborn functions
    class SeabornFallback:
        @staticmethod
        def heatmap(data, ax=None, annot=False, fmt='.2f', cmap='coolwarm', linewidths=0):
            if ax is None:
                ax = plt.gca()
            
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax)
            
            # Add annotations if requested
            if annot:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        text = format(data.iloc[i, j], fmt) if hasattr(data, 'iloc') else format(data[i, j], fmt)
                        ax.text(j, i, text, ha='center', va='center', color='black')
                        
            # Set labels
            if hasattr(data, 'index') and hasattr(data, 'columns'):
                ax.set_xticks(np.arange(data.shape[1]))
                ax.set_yticks(np.arange(data.shape[0]))
                ax.set_xticklabels(data.columns)
                ax.set_yticklabels(data.index)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            return ax
        
        @staticmethod
        def boxplot(x=None, y=None, data=None, ax=None):
            if ax is None:
                ax = plt.gca()
            
            if x is not None and y is not None and data is not None:
                # Group the data by x and create boxplots for each group
                grouped = data.groupby(x)[y]
                positions = range(len(grouped))
                
                box_data = [group[1].values for group in grouped]
                ax.boxplot(box_data, positions=positions)
                
                # Set labels
                ax.set_title(f"{y} by {x}")
                ax.set_ylabel(y)
                ax.set_xlabel(x)
                ax.set_xticks(positions)
                ax.set_xticklabels(grouped.groups.keys())
                
            return ax
        
        @staticmethod
        def pairplot(data, hue=None, palette=None):
            # This is a simplified version that creates a grid of scatter plots
            if hue is not None:
                # Get the numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if hue in numeric_cols:
                    numeric_cols.remove(hue)
                
                n = len(numeric_cols)
                fig, axes = plt.subplots(n, n, figsize=(n*3, n*3))
                
                # Create scatter plots
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        ax = axes[i, j]
                        
                        if i == j:  # Diagonal
                            # Create a histogram
                            ax.hist(data[col1], bins=20)
                            ax.set_title(col1)
                        else:  # Off-diagonal
                            # Create scatter plot with colors by hue
                            for category in data[hue].unique():
                                subset = data[data[hue] == category]
                                ax.scatter(subset[col2], subset[col1], label=str(category), alpha=0.5)
                            
                            if i == n-1:  # Bottom row
                                ax.set_xlabel(col2)
                            if j == 0:  # First column
                                ax.set_ylabel(col1)
                
                # Add a legend to the first plot
                handles, labels = axes[0, 1].get_legend_handles_labels()
                fig.legend(handles, labels, title=hue, loc='upper right')
                
                plt.tight_layout()
            
            return fig
    
    sns = SeabornFallback()

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

FEATURES   = ['ROE','ROA','DebtEq','CurrRatio','AssetTurn']
MODEL_FILE = 'model_RandomForest.pkl'

# Cập nhật tổ chức tệp - cấu trúc thư mục hợp nhất
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PREDICTIONS_FILENAME = 'predictions'

# ---------- Hàm trợ giúp cho đường dẫn tệp tiêu chuẩn hóa ----------
def get_session_path(session_id, file_type, name=None):
    """Tạo đường dẫn tệp tiêu chuẩn hóa cho các sản phẩm phiên"""
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
        # ---------- TỐT ----------
        good = (
            (r.ROE       >= 0.04)   &   # 4 %
            (r.ROA       >= 0.015)  &   # 1.5 %
            (r.DebtEq    <= 50)     &   # nợ ≤ 50 lần vốn
            (r.CurrRatio >= 0.6)
        )
        # ---------- KÉM ----------
        poor = (
            (r.ROE       < 0.015) |
            (r.ROA       < 0.007) |
            (r.DebtEq    > 50)   |      # rất nhiều nợ
            (r.CurrRatio < 0.4)
        )
        if has_AT:   # chưa có AssetTurn, nhưng giữ cho tương lai
            good &= (r.AssetTurn >= 0.10)
            poor |= (r.AssetTurn < 0.03)
        return 2 if good else 0 if poor else 1
    return lab

# ---------- tạo các biểu đồ EDA ----------
def eda_visuals(df: pd.DataFrame, session_id):
    """Tạo các biểu đồ phân tích khám phá dữ liệu và lưu vào thư mục data"""
    eda_dir = get_session_path(session_id, 'eda_dir')
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    cols_to_plot = [col for col in FEATURES if col in df.columns]

    # 1. Ma trận tương quan
    corr = df[cols_to_plot].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Ma trận tương quan giữa các biến")
    plt.savefig(f"{eda_dir}/heatmap_correlation_{session_id}.png", dpi=300)
    plt.close()

    # 2. Biểu đồ hộp theo Rating (nếu có)
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

    # 3. Biểu đồ cặp (nếu có Rating)
    if HAS_SEABORN and 'Rating' in df.columns and len(df) <= 1000:  # Giới hạn kích thước để tránh biểu đồ quá lớn
        sns.pairplot(df[cols_to_plot + ['Rating']], hue='Rating', palette='Set2')
        plt.savefig(f"{eda_dir}/pairplot_by_rating_{session_id}.png", dpi=300)
        plt.close()
    
    return eda_dir

# Trình trợ giúp xử lý lỗi tuyến đường thông thường
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

# Trình trợ giúp tải dữ liệu phân tích
def load_analysis_data(session_id):
    """Tải dữ liệu phân tích chung được sử dụng bởi nhiều tuyến đường"""
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
    
    # Kiểm tra đầy đủ các trường tài chính
    missing_fields, column_mapping = validate_financial_data(df)
    if missing_fields:
        return False, f"❌ Dữ liệu tài chính không đầy đủ: Thiếu {len(missing_fields)} trường: {', '.join(missing_fields[:10])}{'...' if len(missing_fields) > 10 else ''}. File cần có đầy đủ các trường tài chính chuẩn."
    
    # Đổi tên các cột theo định dạng chuẩn
    df = df.rename(columns=column_mapping)
    df = smart_rename(df)

    # tính thêm cột
    if 'AssetTurn' not in df.columns and {'Revenue','Assets'}.issubset(df.columns):
        df['AssetTurn']=df['Revenue']/df['Assets']

    # Kiểm tra xem tất cả các trường bắt buộc cho phân tích có tồn tại không
    feats=[f for f in FEATURES if f in df.columns]
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    if missing_features:
        return False, f"❌ Dữ liệu không hợp lệ cho phân tích: Thiếu các trường bắt buộc: {', '.join(missing_features)}. Vui lòng đảm bảo dữ liệu có đầy đủ các trường: {', '.join(FEATURES)}."
    
    df=df.dropna(subset=feats)

    # Kiểm tra nếu dữ liệu sau khi loại bỏ các dòng có giá trị NaN còn quá ít
    if len(df) < 10:  # Giả sử cần ít nhất 10 dòng để phân tích có ý nghĩa
        return False, f"❌ Dữ liệu không đủ: Sau khi loại bỏ các dòng có giá trị thiếu, chỉ còn {len(df)} dòng dữ liệu. Cần ít nhất 10 dòng để phân tích."

    if 'Rating' not in df.columns:
        df['Rating']=df.apply(build_labeler('AssetTurn'in feats, False),axis=1)

    X,y=df[feats],df['Rating']
    
    # Kiểm tra phân phối nhãn, đảm bảo mỗi lớp có ít nhất một mẫu
    if len(set(y)) < 3:
        missing_classes = set([0, 1, 2]) - set(y)
        return False, f"❌ Dữ liệu không cân bằng: Thiếu mẫu cho các lớp: {missing_classes}. Cần có mẫu cho tất cả 3 lớp (0: Kém, 1: Trung bình, 2: Tốt)."
    
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

        # Lưu báo cáo với tên nhất quán
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

# ==================== ỨNG DỤNG FLASK WEB ====================
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'), 
           static_folder=str(BASE_DIR / 'static'))
app.secret_key = 'finance-analysis-secret-key'
app.config['UPLOAD_FOLDER'] = str(DATA_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@app.route('/dashboard/<path:session_id>')
def dashboard(session_id=None):
    """Hiển thị dashboard khi truy cập /dashboard trực tiếp"""
    # Nếu session_id được cung cấp trong URL, sử dụng nó
    if session_id:
        session['analysis_session_id'] = session_id
        return redirect(url_for('analysis_dashboard', session_id=session_id))
    
    # Nếu không, kiểm tra xem có session_id trong trạng thái phiên không
    session_id = session.get('analysis_session_id')
    if session_id:
        return redirect(url_for('analysis_dashboard', session_id=session_id))
    else:
        flash("Không tìm thấy phiên phân tích đang hoạt động. Vui lòng tải lên một tệp để phân tích.", "info")
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Không có phần tệp')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Không có tệp được chọn')
        return redirect(url_for('index'))
    
    # Kiểm tra định dạng file
    file_ext = Path(file.filename).suffix.lower()
    valid_extensions = ['.csv', '.xlsx', '.xls']
    
    if file_ext not in valid_extensions:
        flash(f"❌ Định dạng file không được hỗ trợ: {file_ext}. Vui lòng tải lên file CSV hoặc Excel (.csv, .xlsx, .xls)", "danger")
        return redirect(url_for('index'))
    
    if file:
        # Lưu tệp đã tải lên trực tiếp vào DATA_DIR
        filename = str(uuid.uuid4()) + file_ext
        file_path = DATA_DIR / filename
        file.save(file_path)
        
        try:
            # Xử lý tệp
            success, result = run_pipeline(file_path)
            
            if not success:
                flash(result, "danger")
                return redirect(url_for('index'))
            
            # Lưu session_id vào session state để phục vụ điều hướng sau này
            session_id = result['session_id']
            session['analysis_session_id'] = session_id  # Lưu trong phiên
            session['current_analysis'] = {
                'session_id': session_id,
                'filename': file.filename,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': FEATURES
            }
            
            # Chuyển hướng đến trang analysis_dashboard với kết quả phân tích trực quan
            return redirect(url_for('analysis_dashboard', session_id=session_id))
        except Exception as e:
            # Xử lý lỗi trong quá trình đọc file
            flash(f"❌ Không thể xử lý file: {str(e)}. Vui lòng kiểm tra nội dung file và đảm bảo đúng định dạng.", "danger")
            return redirect(url_for('index'))

@app.route('/analysis_dashboard/<path:session_id>')
@handle_route_errors
def analysis_dashboard(session_id):
    """Hiển thị dashboard tích hợp với tất cả kết quả phân tích"""
    # Lưu ID phiên trong phiên phía máy chủ
    session['analysis_session_id'] = session_id
    
    # Tải dữ liệu phân tích chung
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
    """Phục vụ tệp từ thư mục dữ liệu"""
    # Xử lý các tệp ma trận nhầm lẫn
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
        
        return f"Không tìm thấy tệp EDA: {eda_name}", 404
    
    # Xử lý các tệp CSV và tệp khác
    elif '.' in filename:
        base, ext = filename.rsplit('.', 1)
        full_filename = f"{base}_{session_id}.{ext}"
    else:
        full_filename = f"{filename}_{session_id}.csv"
    
    if not (DATA_DIR / full_filename).exists():
        return f"Không tìm thấy tệp {full_filename}", 404
        
    return send_from_directory(DATA_DIR, full_filename, as_attachment=False)

@app.route('/download_report/<path:session_id>/<path:model_name>')
def download_report(session_id, model_name):
    report_filename = f'class_report_{session_id}_{model_name}.txt'
    
    if not (DATA_DIR / report_filename).exists():
        flash("Không tìm thấy báo cáo phân loại")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, report_filename, as_attachment=True)

@app.route('/download_csv/<path:session_id>/<path:filename>')
def download_csv(session_id, filename):
    file_path = get_session_path(session_id, filename)
    
    if not file_path.exists():
        flash("Không tìm thấy tệp")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, file_path.name, as_attachment=True)

@app.route('/download_model/<path:session_id>')
@app.route('/download_model/<path:session_id>/<path:model_name>')
def download_model(session_id, model_name=None):
    model_path = get_session_path(session_id, 'model', model_name)
    
    if not model_path.exists():
        flash("Không tìm thấy tệp mô hình")
        return redirect(url_for('index'))
    
    return send_from_directory(DATA_DIR, model_path.name, as_attachment=True)

@app.route('/view_report/<path:session_id>/<path:model_name>')
@handle_route_errors
def view_report(session_id, model_name):
    # Lưu ID phiên trong phiên phía máy chủ
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
    # Lưu ID phiên trong phiên phía máy chủ
    session['analysis_session_id'] = session_id
    
    csv_path = get_session_path(session_id, filename)
    
    if not csv_path.exists():
        if filename == 'predictions_with_prob.csv':
            alt_path = get_session_path(session_id, 'predictions')
            if alt_path.exists():
                csv_path = alt_path
            else:
                flash(f"Không tìm thấy tệp: {filename}")
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
    # Lưu ID phiên trong phiên phía máy chủ
    session['analysis_session_id'] = session_id
    
    # Sử dụng hàm trợ giúp để tải dữ liệu phân tích
    data = load_analysis_data(session_id)
    
    # Trả về mẫu với dữ liệu
    return render_template('view_prediction_results.html',
                          session_id=session_id,
                          models_data=data['models_data'],
                          cm_paths=data['cm_paths'],
                          report_paths=data['report_paths'],
                          stats=data['stats'])

@app.route('/view_eda/<path:session_id>')
@app.route('/view_eda/<path:session_id>/<path:source>')
@handle_route_errors
def view_eda(session_id, source='prediction'):
    """View EDA visualizations with explanations"""
    # Lưu ID phiên trong phiên phía máy chủ
    session['analysis_session_id'] = session_id
    
    # Kiểm tra tồn tại của biểu đồ EDA
    eda_dir = get_session_path(session_id, 'eda_dir')
    
    if not eda_dir.exists() or not any(eda_dir.glob('*.png')):
        flash("Không tìm thấy biểu đồ phân tích dữ liệu", "warning")
        return redirect(url_for('index'))
    
    # Lấy đường dẫn đến các biểu đồ EDA theo thứ tự nhất quán
    eda_files = []
    
    # 1. Tìm biểu đồ tương quan trước
    heatmap_files = list(eda_dir.glob('*heatmap*.png'))
    if heatmap_files:
        eda_files.extend(heatmap_files)
    
    # 2. Sau đó tìm biểu đồ hộp
    boxplot_files = list(eda_dir.glob('*boxplot*.png'))
    if boxplot_files:
        eda_files.extend(boxplot_files)
    
    # 3. Cuối cùng tìm biểu đồ cặp
    pairplot_files = list(eda_dir.glob('*pairplot*.png'))
    if pairplot_files:
        eda_files.extend(pairplot_files)
    
    # Thêm các file còn lại
    for file in eda_dir.glob('*.png'):
        if file not in eda_files:
            eda_files.append(file)
    
    # Chuyển đường dẫn file thành URL cho template
    eda_paths = [f'/file/{session_id}/eda/{Path(p).name}' for p in eda_files]
    
    # Tạo giải thích cho từng loại trực quan hóa
    explanations = {}
    for path in eda_paths:
        if 'heatmap' in path:
            explanations[path] = "Ma trận tương quan thể hiện mối quan hệ tương quan giữa các biến. Số càng gần 1 thể hiện mối quan hệ tương quan càng tích cực, số càng gần -1 thể hiện mối quan hệ tương quan càng tiêu cực."
        elif 'boxplot' in path:
            explanations[path] = "Biểu đồ hộp theo phân loại cho thấy phân phối của từng biến theo nhóm phân loại (Kém/0, Trung bình/1, Tốt/2). Điều này giúp xác định sự khác biệt về giá trị biến giữa các nhóm."
        elif 'pairplot' in path:
            explanations[path] = "Biểu đồ cặp hiển thị mối quan hệ giữa từng cặp biến, phân loại theo nhóm. Giúp xác định các mẫu và xu hướng trong dữ liệu theo nhóm phân loại."
        else:
            explanations[path] = "Biểu đồ phân tích dữ liệu."
    
    return render_template('view_eda.html',
                          session_id=session_id,
                          eda_paths=eda_paths,
                          explanations=explanations,
                          source=source)

# Danh sách đầy đủ các trường tài chính cần có
REQUIRED_FINANCIAL_FIELDS = [
    'ebitdaMargins', 'profitMargins', 'grossMargins', 'operatingCashflow', 
    'revenueGrowth', 'operatingMargins', 'ebitda', 'grossProfits', 
    'freeCashflow', 'currentPrice', 'earningsGrowth', 'currentRatio', 
    'returnOnAssets', 'debtToEquity', 'returnOnEquity', 'totalCash', 
    'totalDebt', 'totalRevenue', 'totalCashPerShare', 'financialCurrency', 
    'revenuePerShare', 'quickRatio', 'quoteType', 'symbol', 
    'enterpriseToRevenue', 'enterpriseToEbitda', 'forwardEps', 
    'sharesOutstanding', 'bookValue', 'trailingEps', 'priceToBook', 
    'heldPercentInsiders', 'enterpriseValue', 'earningsQuarterlyGrowth', 
    'pegRatio', 'forwardPE', 'marketCap'
]

# Hàm kiểm tra đầy đủ các trường tài chính
def validate_financial_data(df):
    """Kiểm tra xem dữ liệu có đầy đủ các trường tài chính cần thiết không"""
    # Chuyển đổi tất cả các tên cột thành chữ thường để so sánh không phân biệt hoa thường
    lower_columns = [col.lower() for col in df.columns]
    
    # Tạo ánh xạ từ tên cột trong tệp dữ liệu đến tên cột chuẩn
    column_mapping = {}
    for field in REQUIRED_FINANCIAL_FIELDS:
        field_lower = field.lower()
        # Tìm tên cột tương ứng trong dữ liệu
        matching_cols = [col for i, col in enumerate(df.columns) 
                         if lower_columns[i] == field_lower or 
                         field_lower in lower_columns[i]]
        if matching_cols:
            column_mapping[field] = matching_cols[0]
    
    # Kiểm tra các trường còn thiếu
    missing_fields = [field for field in REQUIRED_FINANCIAL_FIELDS 
                     if field not in column_mapping]
    
    return missing_fields, column_mapping

# ==================== ĐIỂM NHẬP ====================
if __name__=="__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        run_pipeline(sys.argv[1])      # huấn luyện qua Terminal
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)