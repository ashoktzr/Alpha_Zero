# High-Quality Runs Webhook Setup

This guide explains how to set up your own Google Sheets webhook to capture high-quality cluster runs.

> **Note**: By default, Alpha Zero already logs to the developer's shared sheet. This setup is only needed if you want your own private logging sheet.

## Overview

Alpha Zero automatically logs high-quality cluster runs (Precision ≥ 50%, Coverage ≥ 3%) to Google Sheets. The default webhook is pre-configured - no setup needed!

**If you want your own private logging:**

## Prerequisites

- Google Account
- Google Sheets
- Basic familiarity with Google Apps Script

---

## Step 1: Create Google Sheet

1. Go to [Google Sheets](https://sheets.google.com/)
2. Create a new spreadsheet named "AlphaZero HQ Runs"
3. In the first row, add headers:
   ```
   Timestamp | Profile | Time Period | Timeframe | Features | Scaling Method | TP/SL | Max Lookahead | Cluster IDs | Coverage | Long Precision | Short Precision | Long Win Rate | Short Win Rate
   ```

---

## Step 2: Create Google Apps Script Webhook

1. Open your "AlphaZero HQ Runs" spreadsheet
2. Click **Extensions** → **Apps Script**
3. Delete any existing code and paste the following:

```javascript
function doPost(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    var data = JSON.parse(e.postData.contents);
    
    sheet.appendRow([
      data.timestamp || new Date().toISOString(),
      data.profile || 'Default',
      data.time_period || 'N/A',
      data.timeframe || 'N/A',
      data.features || '',
      data.scaling_method || 'None',
      data.tp_sl || '',
      data.max_lookahead || '',
      data.cluster_ids || '',
      data.coverage || '',
      data.long_precision || '',
      data.short_precision || '',
      data.long_win_rate || '',
      data.short_win_rate || ''
    ]);
    
    return ContentService.createTextOutput(JSON.stringify({
      'status': 'success'
    })).setMimeType(ContentService.MimeType.JSON);
    
  } catch(error) {
    return ContentService.createTextOutput(JSON.stringify({
      'status': 'error',
      'message': error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}
```

4. Click **Deploy** → **New deployment**
5. Click the gear icon ⚙️ next to "Select type" → Choose **Web app**
6. Configure:
   - **Description**: "AlphaZero HQ Runs Webhook"
   - **Execute as**: Me
   - **Who has access**: Anyone
7. Click **Deploy**
8. **Copy the Web app URL** - this is your private webhook URL

---

## Step 3: Configure Your Private Webhook (Optional)

To use your private webhook instead of the default shared one:

1. Navigate to your project's `.streamlit` directory
2. Create or edit `secrets.toml`:

```toml
HQ_RUNS_WEBHOOK = "YOUR_WEBHOOK_URL_HERE"
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY_HERE"  # Optional
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET_HERE"  # Optional
```

3. Replace `YOUR_WEBHOOK_URL_HERE` with your webhook URL from Step 2
4. Save the file

> **⚠️ Important**: The `secrets.toml` file is already gitignored. Never commit your secrets to version control!

---

## Testing

1. Run the clustering pipeline with features
2. Configure TP/SL settings
3. Run forward scan
4. If any clusters meet the criteria (Precision ≥ 50%, Coverage ≥ 3%), check your Google Sheet for a new row

---

## Data Format

The webhook receives data in this format:

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "profile": "My Profile",
  "time_period": "2023-01-01 - 2023-12-31",
  "timeframe": "1h",
  "features": "RSI_14, MACD_12_26_9, BB_20",
  "scaling_method": "Standard",
  "tp_sl": "2.00%, 1.00%",
  "max_lookahead": 100,
  "cluster_ids": "1, 3, 5",
  "coverage": "0.0523, 0.0412, 0.0631",
  "long_precision": "0.5234, 0.5891, 0.5123",
  "short_precision": "0.4821, 0.5234, 0.4987",
  "long_win_rate": "0.5234, 0.5891, 0.5123",
  "short_win_rate": "0.4821, 0.5234, 0.4987"
}
```

---

## Security Notes

- **Webhook URLs are secret**: Anyone with the URL can write to your sheet
- The default shared webhook logs all users' runs to help improve the project
- Use your own private webhook if you want complete privacy
- Consider using Google Sheets API with OAuth for production use

---

For questions or issues, please open an issue on the [GitHub repository](https://github.com/ashoktzr/Alpha_Zero).
