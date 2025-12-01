# Branch Strategy

This repository maintains two parallel branches to serve different deployment scenarios:

## Branches

### `streamlit` (Demo Branch)

**Purpose**: Demo version for users to try and experience the app on Streamlit Community Cloud

**Limitations**:
- Maximum 50,000 data rows
- Maximum 4 features for clustering
- Maximum 60 candles lookahead
- 200MB file upload limit
- float32 precision for memory optimization

**Best for**:
- Trying the app online without local setup
- Quick demos and sharing
- Testing the concept before serious analysis

### `unlimited` (Recommended Branch)

**Purpose**: Full-featured version without artificial constraints - recommended for serious research and analysis

**Features**:
- ✅ Unlimited data rows
- ✅ Unlimited feature selection
- ✅ Unlimited lookahead period
- ✅ 1000MB file upload limit
- ✅ float64 precision for accuracy

**Best for**:
- Serious research and trading strategy development
- Local development with full capabilities
- High-performance analysis
- Large datasets (100k+ rows)
- Advanced feature engineering
- Private cloud deployments

## Switching Branches

### Clone specific branch
```bash
# Clone unlimited branch
git clone -b unlimited https://github.com/ashoktzr/Alpha_Zero.git

# Clone streamlit branch (default)
git clone https://github.com/ashoktzr/Alpha_Zero.git
```

### Switch between branches
```bash
# Switch to unlimited
git checkout unlimited

# Switch to streamlit
git checkout streamlit
```

## Code Differences

The main differences between branches are:

| Feature | Streamlit | Unlimited |
|---------|-----------|-----------|
| Data row limit | 50,000 | None |
| Feature selection limit | 4 | None |
| Max lookahead | 60 candles | None |
| Upload size | 200MB | 1000MB |
| Float precision | float32 | float64 |
| Warning messages | Yes | No |

### Modified Files

The following files have been modified in the `unlimited` branch:

1. **`app/tabs/pipeline_page.py`**

## Deployment

### Streamlit Community Cloud

Use the `streamlit` branch for deployment to Streamlit Community Cloud:

1. Connect your GitHub repository
2. Select `streamlit` branch
3. Set main file path: `app/gui_streamlit.py`
4. Deploy

### Local Development

Use the `unlimited` branch for local development:

```bash
# Clone unlimited branch
git clone -b unlimited https://github.com/ashoktzr/Alpha_Zero.git
cd Alpha_Zero

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app/gui_streamlit.py
```

## Contributing

When contributing:
- Make changes to `unlimited` branch first
- Do NOT add limitations to `unlimited` branch
- Do NOT remove limitations from `streamlit` branch

## Questions?

If you're unsure which branch to use:
- **Just want to try the app?** → Use `streamlit` (demo on cloud)
- **Serious research or analysis?** → Use `unlimited` (recommended)
- **Local development?** → Use `unlimited`
- **Large datasets (>50k rows)?** → Use `unlimited`
- **Advanced analysis?** → Use `unlimited`

> **Note**: The `streamlit` branch is a demo version for users to try and experience the app. For any serious work, use the `unlimited` branch.
