from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import traceback
import uuid

# Default values
DEFAULT_PREDICTIONS_FILENAME = 'predictions'

def create_app(base_dir, data_dir):
    app = Flask(__name__, 
                template_folder=str(base_dir / 'templates'), 
                static_folder=str(base_dir / 'static'))
    
    app.secret_key = 'finance-analysis-secret-key'
    app.config['UPLOAD_FOLDER'] = str(data_dir)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
    
    # Add base_dir and data_dir to app config for access in routes
    app.config['BASE_DIR'] = base_dir
    app.config['DATA_DIR'] = data_dir
    
    # Import run_pipeline function
    from main import run_pipeline

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        """Show dashboard when accessing /dashboard directly"""
        # Check if there's a session_id in the session state
        session_id = session.get('analysis_session_id')
        return render_template('dashboard.html', session_id=session_id)

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
            file_path = data_dir / filename
            file.save(file_path)
            
            # Process file
            success, result = run_pipeline(file_path)
            
            if not success:
                flash(result, "danger")
                return redirect(url_for('index'))
            
            # Lưu session_id vào session state để phục vụ điều hướng sau này
            session_id = result['session_id']
            session['current_analysis'] = {
                'session_id': session_id,
                'filename': file.filename,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': ['ROE','ROA','DebtEq','CurrRatio','AssetTurn','EPS_Growth']
            }
            
            # Chuyển hướng đến trang dashboard với kết quả phân tích trực quan
            return redirect(url_for('analysis_dashboard', session_id=session_id))

    @app.route('/analysis_dashboard/<path:session_id>')
    def analysis_dashboard(session_id):
        """Hiển thị dashboard tích hợp với tất cả kết quả phân tích"""
        try:
            # Kiểm tra sự tồn tại của dữ liệu phân tích
            results_path = data_dir / f'results_{session_id}.csv'
            predictions_path = data_dir / f'predictions_{session_id}.csv'
            
            if not results_path.exists() or not predictions_path.exists():
                flash("Không tìm thấy dữ liệu phân tích cho phiên này", "warning")
                return redirect(url_for('index'))
            
            # Đọc dữ liệu phân tích
            results_df = pd.read_csv(results_path)
            predictions_df = pd.read_csv(predictions_path)
            
            # Chuẩn bị dữ liệu cho template
            models_data = results_df.to_dict('records')
            cm_paths = [f'/file/{session_id}/cm_{model}' for model in results_df['Model']]
            report_paths = [f'/view_report/{session_id}/{model}' for model in results_df['Model']]
            
            # Thống kê cơ bản
            stats = {
                'total_samples': len(predictions_df),
                'good_count': (predictions_df['Rating_pred'] == 2).sum(),
                'avg_count': (predictions_df['Rating_pred'] == 1).sum(),
                'poor_count': (predictions_df['Rating_pred'] == 0).sum(),
                'avg_prob_good': predictions_df['Prob_Good'].mean()
            }
            
            # Kiểm tra xem có biểu đồ EDA không
            eda_dir = data_dir / f'eda_{session_id}'
            has_eda = eda_dir.exists() and any(eda_dir.glob('*.png'))
            
            # Lấy danh sách các biểu đồ EDA nếu có
            eda_paths = []
            if has_eda:
                eda_files = list(eda_dir.glob('*.png'))
                eda_paths = [f'/file/{session_id}/eda/{Path(p).name}' for p in eda_files]
            
            # Trả về dashboard tích hợp
            return render_template('analysis_dashboard.html',
                                session_id=session_id,
                                models_data=models_data,
                                cm_paths=cm_paths,
                                report_paths=report_paths,
                                stats=stats,
                                has_eda=has_eda,
                                eda_paths=eda_paths,
                                analysis_info=session.get('current_analysis', {}),
                                preview_data=predictions_df.head(10))
                                
        except Exception as e:
            traceback.print_exc()
            flash(f"Lỗi khi hiển thị dashboard: {str(e)}", "danger")
            return redirect(url_for('index'))

    @app.route('/file/<path:session_id>/<path:filename>')
    def serve_file(session_id, filename):
        """Serve files from data directory"""
        # Xử lý các tệp confusion matrix
        if filename.startswith('cm_'):
            model_name = filename.replace('cm_', '')
            full_filename = f"cm_{session_id}_{model_name}.png"
            return send_from_directory(data_dir, full_filename, as_attachment=False)
        
        # Xử lý các tệp EDA
        elif filename.startswith('eda/'):
            eda_name = filename.replace('eda/', '')
            eda_dir = data_dir / f'eda_{session_id}'
            
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
        
        if not (data_dir / full_filename).exists():
            return f"File {full_filename} not found", 404
            
        return send_from_directory(data_dir, full_filename, as_attachment=False)

    @app.route('/download_report/<path:session_id>/<path:model_name>')
    def download_report(session_id, model_name):
        report_filename = f'class_report_{session_id}_{model_name}.txt'
        
        if not (data_dir / report_filename).exists():
            flash("Classification report not found")
            return redirect(url_for('index'))
        
        return send_from_directory(data_dir, report_filename, as_attachment=True)

    @app.route('/download_csv/<path:session_id>/<path:filename>')
    def download_csv(session_id, filename):
        file_path = data_dir / f'{filename}_{session_id}.csv'
        
        if not file_path.exists():
            flash("File not found")
            return redirect(url_for('index'))
        
        return send_from_directory(data_dir, file_path.name, as_attachment=True)

    @app.route('/download_model/<path:session_id>')
    def download_model(session_id):
        model_filename = f'model_{session_id}_model_RandomForest.pkl'
        
        if not (data_dir / model_filename).exists():
            flash("Model file not found")
            return redirect(url_for('index'))
        
        return send_from_directory(data_dir, model_filename, as_attachment=True)

    @app.route('/view_report/<path:session_id>/<path:model_name>')
    def view_report(session_id, model_name):
        try:
            # Store the session ID in the server-side session as well
            session['analysis_session_id'] = session_id
            
            report_filename = f'class_report_{session_id}_{model_name}.txt'
            file_path = data_dir / report_filename
            
            if not file_path.exists():
                flash("Không tìm thấy báo cáo phân loại")
                return redirect(url_for('index'))
            
            try:
                with open(file_path, 'r') as f:
                    report_content = f.read()
                
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
                
            except Exception as e:
                flash(f"Lỗi khi đọc file báo cáo: {str(e)}")
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f"Lỗi khi xem báo cáo: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/view_csv/<session_id>/<filename>')
    @app.route('/view_csv/<session_id>', defaults={'filename': DEFAULT_PREDICTIONS_FILENAME})
    def view_csv(session_id, filename):
        try:
            # Store the session ID in the server-side session as well
            session['analysis_session_id'] = session_id
            
            csv_path = data_dir / f'{filename}_{session_id}.csv'
            
            if not csv_path.exists():
                if filename == 'predictions_with_prob.csv':
                    alt_path = data_dir / f'predictions_{session_id}.csv'
                    if alt_path.exists():
                        csv_path = alt_path
                    else:
                        flash(f"File not found: {filename}")
                        return render_template('view_csv.html',
                                        data=None,
                                        filename=filename,
                                        session_id=session_id)
            
            try:
                data = pd.read_csv(csv_path)
                return render_template('view_csv.html',
                                    data=data,
                                    filename=filename,
                                    session_id=session_id)
            except Exception as e:
                flash(f"Error parsing CSV file: {str(e)}")
                return render_template('view_csv.html',
                                    data=None,
                                    filename=filename,
                                    session_id=session_id)
        except Exception as e:
            flash(f"Error loading CSV file: {str(e)}")
            return render_template('view_csv.html',
                                data=None,
                                filename=filename,
                                session_id=session_id)

    @app.route('/training_results/<session_id>')
    def training_results(session_id):
        return redirect(url_for('view_prediction_results', session_id=session_id))

    @app.route('/view_prediction_results/<path:session_id>')
    def view_prediction_results(session_id):
        try:
            # Store the session ID in the server-side session
            session['analysis_session_id'] = session_id
            
            results_path = data_dir / f'results_{session_id}.csv'
            predictions_path = data_dir / f'predictions_{session_id}.csv'
            
            if not results_path.exists() or not predictions_path.exists():
                flash("Không tìm thấy tệp kết quả", "danger")
                return redirect(url_for('index'))
            
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
            
            # Return the templates directly rather than redirecting
            return render_template('view_prediction_results.html',
                                session_id=session_id,
                                models_data=models_data,
                                cm_paths=cm_paths,
                                report_paths=report_paths,
                                stats=stats)
            
        except Exception as e:
            traceback.print_exc()
            flash(f"Lỗi khi xem kết quả dự đoán: {str(e)}", "danger")
            return redirect(url_for('index'))

    @app.route('/view_eda/<path:session_id>')
    @app.route('/view_eda/<path:session_id>/<path:source>')
    def view_eda(session_id, source='prediction'):
        """View EDA visualizations with explanations"""
        try:
            # Store the session ID in the server-side session
            session['analysis_session_id'] = session_id
            
            eda_dir = data_dir / f'eda_{session_id}'
            
            if not eda_dir.exists() or not any(eda_dir.glob('*.png')):
                flash("Không tìm thấy biểu đồ phân tích dữ liệu", "warning")
                return redirect(url_for('index'))
            
            # Get paths to EDA visualizations in a consistent order
            eda_files = []
            # 1. First look for correlation heatmap
            heatmap_files = list(eda_dir.glob('*heatmap*.png'))
            if heatmap_files:
                eda_files.extend(heatmap_files)
            
            # 2. Then look for boxplot
            boxplot_files = list(eda_dir.glob('*boxplot*.png'))
            if boxplot_files:
                eda_files.extend(boxplot_files)
            
            # 3. Finally look for pairplot 
            pairplot_files = list(eda_dir.glob('*pairplot*.png'))
            if pairplot_files:
                eda_files.extend(pairplot_files)
            
            # Add any remaining files
            for file in eda_dir.glob('*.png'):
                if file not in eda_files:
                    eda_files.append(file)
            
            # Convert file paths to URL paths for the template
            eda_paths = [f'/file/{session_id}/eda/{Path(p).name}' for p in eda_files]
            
            # Create explanations for each visualization type
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
        
        except Exception as e:
            traceback.print_exc()
            flash(f"Lỗi khi xem biểu đồ phân tích dữ liệu: {str(e)}", "danger")
            return redirect(url_for('index'))

    return app
