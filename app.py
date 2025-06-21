import gradio as gr
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----- Biến toàn cục để lưu dữ liệu đã upload -----
global_df = None

# ----- Dataset -----
class SocialTrendDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y

# ----- Mô hình DLinear đơn giản -----
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, L]
        out = self.linear(x)
        out = out.permute(0, 2, 1)  # [B, L, C]
        return out

# ----- Tiền xử lý & Train & Dự đoán -----
def prepare_model(raw_df, cutoff_date_str):
    if not all(col in raw_df.columns for col in ["DATE", "THEMES"]):
        raise ValueError("❌ File CSV phải có cột 'DATE' và 'THEMES'!")

    try:
        raw_df['DATE'] = pd.to_datetime(raw_df['DATE'], errors='raise')
    except Exception as e:
        raise ValueError(f"❌ Lỗi định dạng ngày: {str(e)}")

    raw_df['THEMES'] = raw_df['THEMES'].fillna('').apply(lambda x: x.split(';'))
    exploded = raw_df.explode('THEMES')
    exploded['THEMES'] = exploded['THEMES'].str.strip()
    exploded = exploded[exploded['THEMES'] != '']

    daily_theme_counts = exploded.groupby(["DATE", "THEMES"]).size().unstack(fill_value=0)
    top_themes = daily_theme_counts.sum().sort_values(ascending=False).head(10).index.tolist()
    df = daily_theme_counts[top_themes]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    cutoff_date = pd.to_datetime(cutoff_date_str)
    if cutoff_date not in scaled_df.index:
        raise ValueError(f"❌ Ngày cutoff '{cutoff_date_str}' không có trong dữ liệu!")

    future_start_idx = scaled_df.index.get_loc(cutoff_date) - 9
    future_input = scaled_df.iloc[future_start_idx:future_start_idx + 10].values
    future_input = torch.tensor(future_input, dtype=torch.float32).unsqueeze(0)

    seq_len, pred_len = 10, 10
    train_data = scaled_df[scaled_df.index <= cutoff_date].values
    train_dataset = SocialTrendDataset(train_data, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLinear(seq_len, pred_len, scaled_df.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        future_output = model(future_input.to(device)).cpu().squeeze(0).numpy()

    return df, future_output

# ----- Dự đoán top 3 chủ đề hot nhất + phân cụm -----
def forecast_top3_topics_from_global(cutoff_date):
    global global_df
    try:
        if global_df is None:
            return "❌ Vui lòng upload file dữ liệu ở tab đầu tiên trước."
        df = global_df.copy()
        df, future_output = prepare_model(df, cutoff_date)

        summed_predictions = future_output.sum(axis=0)
        top3_indices = summed_predictions.argsort()[-3:][::-1]
        top3_names = [df.columns[idx] for idx in top3_indices]

        df_clean = global_df.copy()
        df_clean["THEMES_CLEAN"] = df_clean["THEMES"].fillna("").str.replace(";", " ", regex=False)

        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df_clean["THEMES_CLEAN"])

        kmeans = KMeans(n_clusters=10, random_state=42)
        df_clean["CLUSTER"] = kmeans.fit_predict(X)

        cluster_labels = {
            0: "Du lịch & Tài nguyên Thiên nhiên",
            1: "An ninh & Hệ thống Tư pháp",
            2: "Y tế & Luật pháp",
            3: "Giáo dục & Giao thông công cộng",
            4: "Khai khoáng, Hạ tầng & Tài nguyên",
            5: "Khủng hoảng & Tài nguyên quý",
            6: "Chính trị & Giáo dục",
            7: "Kinh tế & Phát triển Du lịch",
            8: "Công lý & Quản trị công",
            9: "Tội phạm & Ma túy"
        }

        df_clean["CLUSTER_LABEL"] = df_clean["CLUSTER"].map(cluster_labels)

        top3_labels = []
        for theme in top3_names:
            theme_rows = df_clean[df_clean["THEMES"].str.contains(theme, na=False)]
            if theme_rows.empty:
                label = "Không xác định"
            else:
                label = theme_rows["CLUSTER_LABEL"].mode().iloc[0]
            top3_labels.append(label)

        return "🔥 Top 3 chủ đề hot nhất 10 ngày sau mốc {}:\n\n".format(cutoff_date) + \
               "\n".join(f"{i+1}. {name} ({label})" for i, (name, label) in enumerate(zip(top3_names, top3_labels)))
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# ----- Upload & lưu dữ liệu -----
def upload_and_train(train_file):
    global global_df
    try:
        df = pd.read_csv(train_file.name)
        global_df = df
        return "✅ Đã nhận file: {} dòng, {} cột.".format(*df.shape)
    except Exception as e:
        return f"❌ Lỗi khi đọc file: {str(e)}"

# ----- Giao diện Gradio -----
with gr.Blocks(title="Dự đoán xu hướng chủ đề") as iface:
    gr.Markdown("# 📈 Dự đoán xu hướng chủ đề với Mô hình Timesries")

    with gr.Tab("📂 Upload & Train Data"):
        train_file_input = gr.File(label="Tải lên file CSV (DATE, THEMES, SOURCEURLS)")
        train_output = gr.Textbox(label="Thông tin file", interactive=False)
        train_button = gr.Button("Tải lên & lưu dữ liệu")
        train_button.click(fn=upload_and_train, inputs=train_file_input, outputs=train_output)

    with gr.Tab("🔥 Top 3 chủ đề hot"):
        cutoff_input = gr.Textbox(label="Ngày cutoff (yyyy-mm-dd)", placeholder="2025-05-20")
        top3_output = gr.Textbox(label="Top 3 chủ đề", lines=10)
        top3_button = gr.Button("Dự đoán top 3 chủ đề hot nhất")
        top3_button.click(fn=forecast_top3_topics_from_global, inputs=cutoff_input, outputs=top3_output)

iface.launch()


