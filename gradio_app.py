import os
import joblib
import pandas as pd
from pathlib import Path
import gradio as gr

# Constants and configuration
FEATURES = ['ROE', 'ROA', 'DebtEq', 'CurrRatio', 'AssetTurn', 'EPS_Growth']
MODEL_DIR = Path(__file__).parent / 'data'

# Make sure the model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# Helper function to load the most recent model
def load_latest_model():
    """Load the most recent RandomForest model from the data directory"""
    model_files = list(MODEL_DIR.glob('model_*_RandomForest.pkl'))
    if not model_files:
        return None
    
    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    return joblib.load(latest_model)

# Prediction function
def predict_financial_health(roe, roa, debt_eq, curr_ratio, asset_turn, eps_growth):
    """Make financial health prediction using latest model"""
    try:
        model = load_latest_model()
        if model is None:
            return "No models found. Please upload a pre-trained model to the 'data' directory."
        
        # Create input data frame
        input_data = pd.DataFrame({
            'ROE': [float(roe)],
            'ROA': [float(roa)],
            'DebtEq': [float(debt_eq)],
            'CurrRatio': [float(curr_ratio)],
            'AssetTurn': [float(asset_turn)], 
            'EPS_Growth': [float(eps_growth)]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Create response
        categories = {0: "Kém (Poor)", 1: "Trung bình (Average)", 2: "Tốt (Good)"}
        result = f"<h3>Dự đoán: <span style='color: {'red' if prediction == 0 else 'orange' if prediction == 1 else 'green'};'>{categories[prediction]}</span></h3>"
        
        result += "<h4>Xác suất dự đoán:</h4>"
        result += f"<ul>"
        result += f"<li>Kém: <b>{probabilities[0]:.2%}</b></li>"
        result += f"<li>Trung bình: <b>{probabilities[1]:.2%}</b></li>"
        result += f"<li>Tốt: <b>{probabilities[2]:.2%}</b></li>"
        result += f"</ul>"
        
        # Add interpretation
        result += "<h4>Giải thích:</h4>"
        if prediction == 2:
            result += "<p>Doanh nghiệp có các chỉ số tài chính tốt, khả năng hoạt động ổn định và sinh lời.</p>"
        elif prediction == 1:
            result += "<p>Doanh nghiệp đang ở mức trung bình, có những điểm mạnh nhưng cũng có những rủi ro tiềm ẩn.</p>"
        else:
            result += "<p>Doanh nghiệp đang gặp khó khăn về tài chính, cần cân nhắc các biện pháp cải thiện.</p>"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Create example inputs for the interface
example_inputs = [
    [0.05, 0.02, 2.0, 1.5, 0.7, 0.01],  # Good example
    [0.02, 0.01, 30.0, 0.8, 0.2, 0.005],  # Average example
    [0.01, 0.005, 60.0, 0.3, 0.1, -0.05]  # Poor example
]

# Create Gradio interface
demo = gr.Interface(
    fn=predict_financial_health,
    inputs=[
        gr.Number(label="ROE (Return on Equity)", value=0.05, minimum=0, maximum=1),
        gr.Number(label="ROA (Return on Assets)", value=0.02, minimum=0, maximum=1),
        gr.Number(label="Debt to Equity Ratio", value=2.0, minimum=0),
        gr.Number(label="Current Ratio", value=1.5, minimum=0),
        gr.Number(label="Asset Turnover", value=0.7, minimum=0),
        gr.Number(label="EPS Growth", value=0.01, minimum=-1, maximum=1)
    ],
    outputs=gr.HTML(),
    title="Phân tích tài chính doanh nghiệp",
    description="Nhập các chỉ số tài chính để đánh giá sức khỏe tài chính của doanh nghiệp",
    examples=example_inputs,
    article="""
    <div style="text-align: left; max-width: 800px; margin: 0 auto;">
        <h3>Hướng dẫn sử dụng:</h3>
        <ul>
            <li><b>ROE (Return on Equity):</b> Tỷ suất lợi nhuận trên vốn chủ sở hữu, từ 0 đến 1 (ví dụ: 0.05 = 5%)</li>
            <li><b>ROA (Return on Assets):</b> Tỷ suất lợi nhuận trên tổng tài sản, từ 0 đến 1 (ví dụ: 0.02 = 2%)</li>
            <li><b>Debt to Equity Ratio:</b> Tỷ lệ nợ trên vốn chủ sở hữu</li>
            <li><b>Current Ratio:</b> Tỷ lệ thanh toán hiện hành (tài sản ngắn hạn / nợ ngắn hạn)</li>
            <li><b>Asset Turnover:</b> Vòng quay tài sản (doanh thu / tổng tài sản)</li>
            <li><b>EPS Growth:</b> Tốc độ tăng trưởng lợi nhuận trên mỗi cổ phiếu, từ -1 đến 1 (ví dụ: 0.01 = 1%)</li>
        </ul>
        <h3>Ý nghĩa của kết quả:</h3>
        <ul>
            <li><b>Tốt (Good):</b> Doanh nghiệp có sức khỏe tài chính tốt, các chỉ số ROE, ROA cao, nợ thấp.</li>
            <li><b>Trung bình (Average):</b> Doanh nghiệp có sức khỏe tài chính ở mức trung bình.</li>
            <li><b>Kém (Poor):</b> Doanh nghiệp đang gặp khó khăn về tài chính, ROE và ROA thấp, tỷ lệ nợ cao.</li>
        </ul>
        <p><i>Lưu ý: Đây chỉ là một mô hình dự đoán, kết quả nên được xem xét cùng với các phân tích khác.</i></p>
    </div>
    """
)

# Launch the app if this file is run directly
if __name__ == "__main__":
    demo.launch()
