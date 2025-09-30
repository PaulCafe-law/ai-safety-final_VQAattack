
# AI 安全期末 — ViLT VQA 對抗攻擊 Demo（Notebook）

本專案以 **Jupyter Notebook `ai_safety_final.ipynb`** 展示：

* 使用 **ViLT**（Vision-and-Language Transformer）進行 **VQA（影像問答）** 推理。
* 結合 **MLM（遮蔽語言模型）** 與 **FGSM** 的**影像＋文字聯合對抗擾動**，降低模型對正解的信心並抑制 MLM 從填空中「復原」答案。
* 範例輸入影像為 `key.jpg`，執行後會輸出對抗影像 `adv_key.jpg`，並列出每回合的損失與對抗問題。

> 若你只想看/跑 Notebook，打開 `ai_safety_final.ipynb` 按順序執行所有 cell 即可。

---

## 目錄結構

```
AI安全/
├── 原本paper/                # 參考資料
├── ai_safety_final.ipynb      # 主程式（Jupyter Notebook）
├── key.jpg                     # 測試影像（輸入）
├── 商務計劃.pdf               # 專案提案的文件（可忽略）
└── 期末專題提案.pptx          # 與專案無關的簡報（可忽略）
```

---

## 環境需求

* Python 3.8+
* 建議有 GPU（無 GPU 亦可執行，速度較慢）
* 主要套件：

  * `torch`, `torchvision`
  * `transformers`
  * `Pillow`
  * `nltk`

安裝指令：

```bash
pip install torch torchvision pillow transformers nltk
```

首次執行 Notebook 時，程式會自動下載：

* ViLT 權重：`dandelin/vilt-b32-finetuned-vqa`（VQA）、`dandelin/vilt-b32-mlm`（MLM）
* NLTK WordNet 資源（用於同義詞替換）

---

## 快速開始（Notebook）

1. 將 `key.jpg` 放在與 Notebook 同一層資料夾（已就緒）。
2. 開啟並執行 `ai_safety_final.ipynb`：

   * 依序執行安裝/匯入/下載資源的 cell。
   * 執行主攻擊流程，預設使用：

     ```python
     q = "Where is the key roughly located?"
     correct = "table"
     M = 6      # 迭代回合
     eps = 0.02 # FGSM 步長
     ```
3. 成功後會在專案根目錄輸出 `adv_key.jpg`，終端（或 Notebook 的輸出格）會顯示每回合：

   * `p_corr`：模型對正解類別的機率
   * `loss_lat`：潛在特徵相似度損失（希望越低越好）
   * `loss_anti`：MLM 反復原損失
   * 以及目前的對抗問題 `adv_q`（包含同義詞微擾）

---

## 重要參數

* **`M`**（迭代次數）：越大可能越強，但也更花時間。
* **`eps`**（FGSM 步長）：越大擾動越明顯；太大會有可見雜訊。
* **`correct`**（正解）：**必須在 ViLT VQA 的答案集合中**。若程式拋出錯誤，請改用集合內更常見的單詞（如 `"table"`, `"floor"`, `"kitchen"` 等）。

---

## 常見問題

* **ValueError：答案不在支援清單**
  ViLT VQA 為封閉詞彙分類器，請改用集合內的近義詞/常見詞；或先跑一次乾淨前向，觀察 top-k 輸出後挑選。

* **找不到 `[MASK]`**
  確認產生遮蔽句的函式會使用 `processor.tokenizer.mask_token`；若你改動了 prompt 模板，需保證真的插入 `[MASK]`。

* **影像看起來太花**
  降低 `eps`、或減少 `M`。如需更平滑的對抗擾動，可自行在 Notebook 內加入後處理（非本範例重點）。

* **速度慢**
  建議使用 GPU。Notebook 已盡量在不需梯度處關閉 `grad`。

---


## 研究與倫理聲明

此 Notebook 僅供研究與教學之用，用於展示多模態模型的脆弱性與防禦研究動機。**請勿用於任何惡意或不當用途。**

---

## 授權

 MIT LICENSE

---

