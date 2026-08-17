# Invoice Line Item Scraper

Browser automation script to extract invoice line items from the platform.

## Prerequisites

- **Node.js** 
- **npm**

## Quick Start

### 1. Install dependencies

```bash
cd data_scraper
sudo apt install npm
install npm
```

This will install Playwright and download the Chromium browser.

### 3. Run the scraper


```
node scrape_invoices.js
```

**Manual login**
- The browser will open, log in with username and password
- Press Enter in the terminal when you're on the "completed" page for the client. 


### 4. Output

The script creates `invoice_line_items.csv` with columns:

| Invoice fields | Line item fields |
|----------------|------------------|
| Transaction ID | Line Number |
| Type | Line Description |
| File Name | Line Quantity |
| Invoice Date | Line Net Amount |
| Supplier | Line Tax Amount |
| Invoice Total | Line Gross Amount |
| Invoice Category | |
| Expense Category | |
| Gross/Net/Tax amounts | |
| Currency | |
| Vendor Name | |
| Due Date | |

Each row represents one line item. Invoices with 3 line items will have 3 rows (with the invoice header data repeated).


