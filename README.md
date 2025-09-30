# vqattack_full — ViLT 影像問答對抗攻擊 Demo

這個專案展示如何用 **ViLT**（Vision-and-Language Transformer）在 **VQA（影像問答）** 與 **MLM（遮蔽語言模型）** 的雙頭上，對影像與問題做聯合式對抗擾動，降低模型對正確答案的信心並抑制 MLM 從填空中「復原」正解。

## 功能概述

* 使用 **`ViltForQuestionAnswering`** 推理原始影像＋問題，並在最後隱層對齊上做特徵攻擊（cosine 相似度）。
* 使用 **`ViltForMaskedLM`** 對「包含 [MASK] 的 prompt」做 MLM 推理，針對正解的 token 機率加以抑制（anti-recovery）。
* 對影像做 **FGSM** 更新；每數步對文字做**同義詞替換**（NLTK WordNet）以引入細微語義擾動。
* 產出對抗影像 `adv_key.jpg` 與最終對抗問題（終端列印）。

---

## 環境需求

* Python 3.8+
* PyTorch（建議有 GPU）
* 主要套件：`transformers`, `torchvision`, `Pillow`, `nltk`

安裝範例：

```bash
pip install torch torchvision pillow transformers nltk
```

> 第一次執行會自動下載 ViLT 權重與 NLTK 的 WordNet 資源。

---

## 主要專案結構

```
.
├─ ai_safety_final.ipynb      # Python 程式（可命名為此檔）
├─ key.jpg                    # 測試影像（輸入）
└─ adv_key.jpg                # 對抗影像（輸出，程式執行後產生）
```

---

## 主要檔案說明

* `vqattack.py`（本文中的程式）：

  * 讀取 ViLT VQA 模型 `dandelin/vilt-b32-finetuned-vqa`
  * 讀取 ViLT MLM 模型 `dandelin/vilt-b32-mlm`
  * 攻擊核心函式：`vqattack_full(image, question, correct_ans, M=10, eps=0.01)`

---

## 使用方式

1. 準備輸入影像 `key.jpg`（或改成你自己的檔名）。
2. 執行：

```bash
python vqattack.py
```

或在 Notebook 內直接執行 `__main__` 區塊。

執行後會：

* 在終端輸出每次迭代的 `p_corr`（模型對正解的機率）、`loss_lat`、`loss_anti` 與目前問題文本。
* 於專案根目錄輸出 `adv_key.jpg`。
* 列印「最終對抗問題」。

---

## 參數與介面

```python
adv_img, adv_q = vqattack_full(
    image,            # PIL.Image (RGB)
    question,         # str：原始問題
    correct_ans,      # str：正解答案（必須存在於 ViLT VQA 的答案集合）
    M=6,              # 迭代次數
    eps=0.02          # FGSM 步長（每步像素更新尺度）
)
```

* **`M`**：攻擊迭代回合數。越大可能越有效，但也更花時間。
* **`eps`**：每次 FGSM 的步長，過大可能造成可見雜訊；過小可能效果不足。
* **`correct_ans` 必須在模型的 `id2label`/`label2id`** 中，例如示例用的 `"table"`。若不在集合中，程式會丟出：

  ```
  ValueError: 答案 'xxx' 不在模型支持的答案列表中。
  ```

  你可以把正解改成集合內的近義詞（例如 `"on the table"` → `"table"` / `"on table"`，需與模型答案模板一致）。

---

## 攻擊流程與設計要點

1. **乾淨輸入編碼**
   使用 `ViltProcessor` 取得 `pixel_values` 與 `input_ids`，跑一次 VQA 取得**最後隱層特徵**當作「乾淨特徵」`clean_feat`。

2. **對抗變量初始化**

   * 影像張量 `pixel` 設為 `requires_grad_(True)`，以便後續對像素做 FGSM。
   * 問題 `input_ids` 從原始 `question` 取得。

3. **答案映射**

   * 透過 `vqa_model.config.id2label` / `label2id` 取得 `correct_ans` 的類別 id。

4. **前向與損失**（每次迭代）

   * **VQA 前向**：取得 `out.logits` 與 `hidden_states`，計算：

     * `p_corr`：模型對正解類別的機率。
     * `loss_latent`：乾淨與擾動之**最後隱層特徵均值**的 cosine 相似度（希望降低相似、造成語意偏離）。

   * **MLM 反復原**：

     * 以 `make_masked_prompt(correct_ans)` 生成帶 `[MASK]` 的句子，如：
       `In this image, the key appears [MASK] [MASK] ...`
     * 用 `ViltForMaskedLM` 推理，取第一個 `[MASK]` 的 logits，對應正解 token 的機率 `p_recover` 做 **負對數** 懲罰：`loss_anti = -log(p_recover)`

   * **總損失**：`total_loss = loss_latent + loss_anti`
     反向傳播後，僅用於更新 **影像像素**（FGSM）。

5. **影像更新（FGSM）**
   `pixel = (pixel + eps * sign(grad)).clamp(0, 1)`

6. **文字擾動（每 3 步）**
   使用 **WordNet** 找到第一個可替換的同義詞，對 `adv_q` 進行一次替換，並重新編碼成 `input_ids`。

---

## 範例（程式內建）

```python
img = Image.open("key.jpg").convert("RGB")
q = "Where is the key roughly located?"
correct = "table"

adv_img, adv_q = vqattack_full(img, q, correct, M=6, eps=0.02)
from torchvision.transforms.functional import to_pil_image
to_pil_image(adv_img[0]).save("adv_key.jpg")
print("最終對抗問題：", adv_q)
```

---

## 常見問題（FAQ）

**Q1. 跑起來說 `答案 'xxx' 不在模型支持的答案列表中`？**
A：ViLT VQA 是**封閉詞彙**分類器，答案必須在它的 label set 內。可嘗試用更短或更常見的單詞（如 `"table"`, `"floor"`, `"kitchen"`），或先用乾淨前向查看 `top-k` 預測以決定可用詞。

**Q2. `找不到任何 mask token`？**
A：檢查 `make_masked_prompt()` 是否正確使用 `processor.tokenizer.mask_token`；或你改了 prompt 模板導致沒有插入 `[MASK]`。

**Q3. 影像太花、肉眼可見？**
A：將 `eps` 調小、或減少 `M`；也可在每步加入小的平滑（非本範例重點，需自行擴充）。

**Q4. 推理很慢？**
A：建議使用 GPU，並在 `torch.no_grad()` 區塊中避免不必要的梯度計算（本程式已儘量避免）。

---

## 研究與倫理聲明

此程式僅供**研究與教學**使用，演示多模態模型的脆弱性與防禦研究動機，**請勿用於任何惡意用途**。若要在論文或報告中使用，請同時說明你的資料來源與使用情境。

---

## 授權

MIT License

---

如果你要把這份 README 直接放在 GitHub，我可以再幫你微調專案名稱、加上徽章、範例結果圖與執行 GIF。
