/**
Scraper to get data the platform
 */

const { chromium } = require('playwright');
const fs = require('fs');

const CONFIG = {
  baseUrl: 'https://[....]',
  transactionsUrl: 'https://[....]',
  outputFile: 'completed_invoice_line_items.csv',
  headless: false,
  delayBetweenInvoices: 1000,
  // NEW: Save a checkpoint every N invoices (set to 0 to disable)
  checkpointInterval: 20,
};

class InvoiceScraper {
  constructor() {
    this.browser = null;
    this.page = null;
  }

  async init() {
    console.log('🚀 Starting browser...');
    this.browser = await chromium.launch({ 
      headless: CONFIG.headless,
      slowMo: 50,
    });
    this.page = await this.browser.newPage();
    this.page.setDefaultTimeout(20000);
  }

  async login() {
    console.log('🔐 Opening login page...');
    console.log('\n⚠️  Please log in manually in the browser.');
    console.log('   1. Log in');
    console.log('   2. Select your client');
    console.log('   3. Go to Costs page');
    console.log('   4. Click the Completed tab');
    console.log('   5. Press Enter here...\n');
    
    await this.page.goto(CONFIG.baseUrl);
    
    await new Promise(resolve => {
      process.stdin.once('data', resolve);
    });
    
    this.currentListUrl = this.page.url();
    console.log(`✅ Continuing from: ${this.currentListUrl}`);
  }

  async getInvoicesOnCurrentPage() {
    await this.page.waitForSelector('table tbody tr', { timeout: 15000 });
    
    const invoices = await this.page.$$eval('table tbody tr', rows => {
      return rows.map(row => {
        const cells = row.querySelectorAll('td');
        const link = row.querySelector('a');
        
        if (!link) return null;
        
        return {
          transactionId: link.textContent?.trim() || '',
          type: cells[2]?.textContent?.trim() || '',
          fileName: cells[3]?.textContent?.trim() || '',
          invoiceDate: cells[4]?.textContent?.trim() || '',
          supplier: cells[5]?.textContent?.trim() || '',
          total: cells[6]?.textContent?.trim() || '',
          tax: cells[7]?.textContent?.trim() || '',
          category: cells[8]?.textContent?.trim() || '',
        };
      }).filter(inv => inv !== null && inv.transactionId);
    });

    return invoices;
  }

  // ============================================================
  // CHANGED: New method to extract header info (extracted for clarity)
  // ============================================================
  async extractHeaderInfo() {
    let headerInfo = {};
    try {
      headerInfo = await this.page.evaluate(() => {
        const getValue = (labelText) => {
          const allElements = document.querySelectorAll('*');
          for (const el of allElements) {
            if (el.childNodes.length === 1 && 
                el.textContent?.trim() === labelText) {
              const parent = el.closest('div');
              if (parent) {
                const input = parent.querySelector('input, select');
                if (input) return input.value || '';
                const valueDiv = parent.querySelector('div:not(:first-child)');
                if (valueDiv) return valueDiv.textContent?.trim() || '';
              }
            }
          }
          return '';
        };

        return {
          expenseCategory: getValue('Expense category'),
          grossAmount: getValue('Gross amount'),
          netAmount: getValue('Net amount'),
          taxAmount: getValue('Total tax amount'),
          currency: getValue('Currency'),
          vendorName: getValue('Vendor name'),
        };
      });
    } catch (e) {
      console.log(`    ⚠️ Header info error: ${e.message}`);
    }
    return headerInfo;
  }

  // ============================================================
  // NEW: Method to scroll the right panel to reveal Line items
  // ============================================================
  async scrollRightPanelToBottom() {
    // Try multiple strategies to find and scroll the right panel
    const scrolled = await this.page.evaluate(() => {
      // Strategy 1: Look for common panel class patterns
      const panelSelectors = [
        '[class*="detail-panel"]',
        '[class*="right-panel"]',
        '[class*="sidebar"]',
        '[class*="panel"][class*="right"]',
        // Strategy 2: Find scrollable container near "Detail" tab
        'div[class*="scroll"]',
      ];
      
      for (const selector of panelSelectors) {
        const panel = document.querySelector(selector);
        if (panel && panel.scrollHeight > panel.clientHeight) {
          panel.scrollTo(0, panel.scrollHeight);
          return true;
        }
      }
      
      // Strategy 3: Find the "Line items" text and scroll it into view
      const lineItemsLabel = [...document.querySelectorAll('*')].find(
        el => el.textContent?.trim() === 'Line items' && el.childNodes.length <= 2
      );
      if (lineItemsLabel) {
        lineItemsLabel.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return true;
      }
      
      // Strategy 4: Find any scrollable parent of "Vendor & buyer" section
      const vendorSection = [...document.querySelectorAll('*')].find(
        el => el.textContent?.trim() === 'Vendor & buyer'
      );
      if (vendorSection) {
        let parent = vendorSection.parentElement;
        while (parent) {
          if (parent.scrollHeight > parent.clientHeight) {
            parent.scrollTo(0, parent.scrollHeight);
            return true;
          }
          parent = parent.parentElement;
        }
      }
      
      return false;
    });
    
    if (scrolled) {
      await this.page.waitForTimeout(500);
    }
    
    return scrolled;
  }

  // ============================================================
  // NEW: Method to click the Line item button and open the panel
  // ============================================================
  async openLineItemsPanel() {
    try {
      // The "Line item" button is in the "Line items" section on the right panel
      // It shows a badge with the count (e.g., "1" or "2")
      
      // Strategy 1: Click element containing "Line item" text (not "Line items" header)
      const clicked = await this.page.evaluate(() => {
        // Find all elements with "Line item" text
        const elements = [...document.querySelectorAll('*')].filter(el => {
          const text = el.textContent?.trim();
          // Match "Line item" but avoid the "Line items" header
          return text === 'Line item' || 
                 (text?.startsWith('Line item') && !text.startsWith('Line items'));
        });
        
        // Find the clickable one (usually has a badge/count nearby)
        for (const el of elements) {
          const parent = el.closest('[class*="clickable"], [class*="button"], [role="button"], button, a') ||
                        el.closest('div[class]');
          if (parent) {
            parent.click();
            return true;
          }
        }
        
        // Fallback: just click the element itself
        if (elements.length > 0) {
          elements[0].click();
          return true;
        }
        
        return false;
      });
      
      if (!clicked) {
        // Fallback: Try Playwright's text selector
        await this.page.click('text=/^Line item$/');
      }
      
      // Wait for the bottom panel to slide up
      await this.page.waitForTimeout(1000);
      
      return true;
    } catch (e) {
      console.log(`    ⚠️ Could not click Line item button: ${e.message}`);
      return false;
    }
  }

  // ============================================================
  // NEW: Method to extract line items from the bottom panel table
  // FIXED: Auto-detects column layout (with or without checkbox)
  // ============================================================
  async extractLineItemsFromPanel() {
    // First, scroll the bottom panel to ensure all rows are loaded
    await this.page.evaluate(() => {
      // Find the Line Item panel/container and scroll it
      const panels = document.querySelectorAll('div');
      for (const panel of panels) {
        const text = panel.textContent || '';
        if (text.includes('Line Item') && text.includes('Description') && text.includes('Gross amount')) {
          // This looks like the line items panel
          if (panel.scrollHeight > panel.clientHeight) {
            panel.scrollTo(0, panel.scrollHeight);
          }
          break;
        }
      }
    });
    
    await this.page.waitForTimeout(300);
    
    return await this.page.evaluate(() => {
      const items = [];
      
      // Find the Line Item table
      let targetTable = null;
      
      // Strategy 1: Find container with "Line Item" header
      const allDivs = document.querySelectorAll('div');
      for (const div of allDivs) {
        for (const child of div.children) {
          if (child.textContent?.trim() === 'Line Item') {
            targetTable = div.querySelector('table');
            if (targetTable) break;
          }
        }
        if (targetTable) break;
      }
      
      // Strategy 2: Find table near "Line Item" text
      if (!targetTable) {
        const lineItemHeaders = [...document.querySelectorAll('*')].filter(
          el => el.textContent?.trim() === 'Line Item' && 
                el.tagName !== 'TABLE' && 
                el.tagName !== 'TBODY'
        );
        
        for (const header of lineItemHeaders) {
          let parent = header.parentElement;
          for (let i = 0; i < 5 && parent; i++) {
            const table = parent.querySelector('table');
            if (table) {
              targetTable = table;
              break;
            }
            parent = parent.parentElement;
          }
          if (targetTable) break;
        }
      }
      
      // Strategy 3: Find visible table with Description header
      if (!targetTable) {
        const tables = document.querySelectorAll('table');
        for (const table of tables) {
          const headerText = table.querySelector('thead, tr:first-child')?.textContent?.toLowerCase() || '';
          if (headerText.includes('description') && headerText.includes('quantity')) {
            const rect = table.getBoundingClientRect();
            if (rect.top > 100 && rect.height > 0) {
              targetTable = table;
              break;
            }
          }
        }
      }
      
      if (!targetTable) {
        return items;
      }
      
      // ============================================================
      // FIXED: Detect column layout from header row
      // ============================================================
      const headerRow = targetTable.querySelector('thead tr, tr:first-child');
      const headerCells = headerRow ? headerRow.querySelectorAll('th, td') : [];
      
      // Find column indices by header text
      let colIndices = {
        no: -1,
        description: -1,
        quantity: -1,
        netAmount: -1,
        taxAmount: -1,
        grossAmount: -1,
      };
      
      headerCells.forEach((cell, idx) => {
        const text = cell.textContent?.trim().toLowerCase() || '';
        if (text.includes('no.') || text === 'no') colIndices.no = idx;
        else if (text.includes('description')) colIndices.description = idx;
        else if (text.includes('quantity')) colIndices.quantity = idx;
        else if (text.includes('net')) colIndices.netAmount = idx;
        else if (text.includes('tax')) colIndices.taxAmount = idx;
        else if (text.includes('gross')) colIndices.grossAmount = idx;
      });
      
      // Fallback if headers not found: assume no checkbox column
      if (colIndices.description === -1) {
        colIndices = { no: 0, description: 1, quantity: 2, netAmount: 3, taxAmount: 4, grossAmount: 5 };
      }
      
      // Extract rows
      const rows = targetTable.querySelectorAll('tbody tr');
      
      rows.forEach((row, index) => {
        const cells = row.querySelectorAll('td');
        if (cells.length < 3) return;
        
        const getTextOrInput = (cell) => {
          if (!cell) return '';
          const input = cell.querySelector('input');
          if (input) return input.value || '';
          return cell.textContent?.trim() || '';
        };
        
        const description = getTextOrInput(cells[colIndices.description]);
        
        // Skip empty rows or header-like rows
        if (!description || description.toLowerCase() === 'description') return;
        
        // Extract line number - handle dropdown format "1 ▼"
        let lineNum = getTextOrInput(cells[colIndices.no]);
        lineNum = lineNum.replace(/[▼▲\s]/g, '').trim() || String(index + 1);
        
        items.push({
          lineNumber: lineNum,
          description: description,
          quantity: getTextOrInput(cells[colIndices.quantity]),
          netAmount: getTextOrInput(cells[colIndices.netAmount]),
          taxAmount: getTextOrInput(cells[colIndices.taxAmount]),
          grossAmount: getTextOrInput(cells[colIndices.grossAmount]),
        });
      });
      
      return items;
    });
  }

  // ============================================================
  // CHANGED: Complete rewrite of scrapeInvoiceDetail
  // Now includes: scroll → click Line item → extract from panel
  // ============================================================
  async scrapeInvoiceDetail(transactionId) {
    console.log(`  📄 ${transactionId}`);
    
    // Click on the transaction ID link
    await this.page.click(`a:has-text("${transactionId}")`);
    
    // Wait for the detail page to load
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForTimeout(1000);

    // Extract header info from the right panel
    const headerInfo = await this.extractHeaderInfo();

    // ========== NEW LINE ITEMS EXTRACTION FLOW ==========
    let lineItems = [];
    try {
      // Step 1: Scroll the right panel down to reveal "Line items" section
      console.log(`    📜 Scrolling to Line items...`);
      await this.scrollRightPanelToBottom();
      
      // Step 2: Click the "Line item" button to open the bottom panel
      console.log(`    🖱️  Clicking Line item button...`);
      const panelOpened = await this.openLineItemsPanel();
      
      if (panelOpened) {
        // Step 3: Wait for the panel and table to fully render
        await this.page.waitForTimeout(800);
        
        // Wait for the table to appear in the bottom panel
        try {
          await this.page.waitForSelector('table:has(th:has-text("Description"))', { timeout: 5000 });
        } catch (e) {
          // Fallback: wait a bit more
          await this.page.waitForTimeout(500);
        }
        
        // Step 4: Extract line items from the bottom panel table
        lineItems = await this.extractLineItemsFromPanel();
        
        // Debug: if no line items found, log table info
        if (lineItems.length === 0) {
          const debugInfo = await this.page.evaluate(() => {
            const tables = document.querySelectorAll('table');
            return Array.from(tables).map((t, i) => ({
              index: i,
              rows: t.querySelectorAll('tbody tr').length,
              headerText: t.querySelector('thead, tr:first-child')?.textContent?.slice(0, 100),
              visible: t.getBoundingClientRect().height > 0,
            }));
          });
          console.log(`    🔍 Debug - Tables found: ${JSON.stringify(debugInfo)}`);
        }
      }
    } catch (e) {
      console.log(`    ⚠️ Line items error: ${e.message}`);
    }
    // ====================================================

    console.log(`    ✅ ${lineItems.length} line items`);

    return {
      transactionId,
      ...headerInfo,
      lineItems,
    };
  }

  // ============================================================
  // CHANGED: navigateBackToList - Use Costs link then Completed tab
  // This is more reliable than trying to click the breadcrumb
  // ============================================================
  async navigateBackToList() {
    // NEW: Click "Costs" in the breadcrumb first
    try {
      await this.page.click('text=Costs', { timeout: 5000 });
      await this.page.waitForLoadState('networkidle');
    } catch (e) {
      // If "Costs" text click fails, try the sidebar icon
      console.log('    Trying sidebar navigation...');
      await this.page.click('[class*="sidebar"] >> nth=0').catch(() => {});
      await this.page.waitForLoadState('networkidle');
    }
    
    // Then click the "Completed" tab
    await this.page.click('text=Completed');
    await this.page.waitForLoadState('networkidle');
    
    // Wait for the table to appear
    await this.page.waitForSelector('table tbody tr', { timeout: 15000 });
    await this.page.waitForTimeout(500);
  }

  async goToNextPage() {
    try {
      const buttons = await this.page.$$('button');
      
      for (const button of buttons) {
        const text = await button.textContent();
        if (text?.trim() === '>') {
          const isDisabled = await button.isDisabled();
          if (!isDisabled) {
            await button.click();
            await this.page.waitForLoadState('networkidle');
            await this.page.waitForTimeout(500);
            this.currentListUrl = this.page.url();
            return true;
          }
        }
      }
    } catch (e) {
      console.log(`    Pagination error: ${e.message}`);
    }
    return false;
  }

  async scrapeAllInvoices() {
    let currentPage = 1;
    let allResults = [];
    let totalProcessed = 0;  // NEW: Track total invoices processed
    let checkpointNum = 1;   // NEW: Checkpoint counter
    
    try {
      const paginationText = await this.page.textContent('text=/\\d+-\\d+ of \\d+/');
      console.log(`\n📋 Pagination: ${paginationText}\n`);
    } catch (e) {
      console.log('\n📋 Starting scrape...\n');
    }

    while (true) {
      console.log(`\n📄 Page ${currentPage}:`);
      
      const invoices = await this.getInvoicesOnCurrentPage();
      console.log(`   Found ${invoices.length} invoices`);
      
      for (let i = 0; i < invoices.length; i++) {
        const invoice = invoices[i];
        console.log(`\n[Page ${currentPage}, Invoice ${i + 1}/${invoices.length}]`);
        
        try {
          const details = await this.scrapeInvoiceDetail(invoice.transactionId);
          
          allResults.push({
            ...invoice,
            ...details,
          });
          
          // Go back to the list
          await this.navigateBackToList();
          
        } catch (e) {
          console.log(`    ❌ Error: ${e.message}`);
          allResults.push({ ...invoice, lineItems: [], error: e.message });
          
          // ============================================================
          // CHANGED: Simplified recovery - just use the same navigation
          // ============================================================
          try {
            await this.navigateBackToList();
          } catch (recoveryError) {
            console.log('    Recovery failed, reloading transactions page...');
            await this.page.goto(CONFIG.transactionsUrl);
            await this.page.waitForLoadState('networkidle');
            await this.page.click('text=Completed').catch(() => {});
            await this.page.waitForTimeout(1000);
          }
        }
        
        // NEW: Increment counter and save checkpoint if needed
        totalProcessed++;
        if (CONFIG.checkpointInterval > 0 && totalProcessed % CONFIG.checkpointInterval === 0) {
          this.saveCheckpoint(allResults, checkpointNum);
          checkpointNum++;
        }
        
        await this.page.waitForTimeout(CONFIG.delayBetweenInvoices);
      }

      const hasNextPage = await this.goToNextPage();
      if (!hasNextPage) {
        console.log('\n✅ Reached last page');
        break;
      }
      currentPage++;
    }

    return allResults;
  }

  exportToCSV(invoices) {
    console.log('\n💾 Exporting to CSV...');
    
    const headers = [
      'Transaction ID',
      'Type',
      'File Name',
      'Invoice Date',
      'Supplier',
      'Total',
      'Tax',
      'Category',
      'Expense Category',
      'Gross Amount (Header)',
      'Net Amount (Header)',
      'Tax Amount (Header)',
      'Currency',
      'Vendor Name',
      'Line Number',
      'Line Description',
      'Line Quantity',
      'Line Net Amount',
      'Line Tax Amount',
      'Line Gross Amount',
    ];

    const rows = [headers.join(',')];

    for (const invoice of invoices) {
      if (invoice.lineItems && invoice.lineItems.length > 0) {
        for (const line of invoice.lineItems) {
          rows.push([
            this.esc(invoice.transactionId),
            this.esc(invoice.type),
            this.esc(invoice.fileName),
            this.esc(invoice.invoiceDate),
            this.esc(invoice.supplier),
            this.esc(invoice.total),
            this.esc(invoice.tax),
            this.esc(invoice.category),
            this.esc(invoice.expenseCategory),
            this.esc(invoice.grossAmount),
            this.esc(invoice.netAmount),
            this.esc(invoice.taxAmount),
            this.esc(invoice.currency),
            this.esc(invoice.vendorName),
            this.esc(line.lineNumber),
            this.esc(line.description),
            this.esc(line.quantity),
            this.esc(line.netAmount),
            this.esc(line.taxAmount),
            this.esc(line.grossAmount),
          ].join(','));
        }
      } else {
        rows.push([
          this.esc(invoice.transactionId),
          this.esc(invoice.type),
          this.esc(invoice.fileName),
          this.esc(invoice.invoiceDate),
          this.esc(invoice.supplier),
          this.esc(invoice.total),
          this.esc(invoice.tax),
          this.esc(invoice.category),
          this.esc(invoice.expenseCategory),
          this.esc(invoice.grossAmount),
          this.esc(invoice.netAmount),
          this.esc(invoice.taxAmount),
          this.esc(invoice.currency),
          this.esc(invoice.vendorName),
          '', '', '', '', '', '',
        ].join(','));
      }
    }

    fs.writeFileSync(CONFIG.outputFile, rows.join('\n'), 'utf8');
    console.log(`✅ Saved ${rows.length - 1} rows to ${CONFIG.outputFile}`);
  }

  esc(value) {
    if (value === null || value === undefined) return '';
    const str = String(value);
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  }

  // ============================================================
  // NEW: Save a checkpoint copy of the data
  // ============================================================
  saveCheckpoint(invoices, checkpointNum) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const checkpointFile = CONFIG.outputFile.replace('.csv', `_checkpoint_${checkpointNum}_${timestamp}.csv`);
    
    console.log(`\n💾 Saving checkpoint (${invoices.length} invoices) → ${checkpointFile}`);
    
    // Reuse the same CSV generation logic
    const headers = [
      'Transaction ID', 'Type', 'File Name', 'Invoice Date', 'Supplier',
      'Total', 'Tax', 'Category', 'Expense Category', 'Gross Amount (Header)',
      'Net Amount (Header)', 'Tax Amount (Header)', 'Currency', 'Vendor Name',
      'Line Number', 'Line Description', 'Line Quantity', 'Line Net Amount',
      'Line Tax Amount', 'Line Gross Amount',
    ];

    const rows = [headers.join(',')];

    for (const invoice of invoices) {
      if (invoice.lineItems && invoice.lineItems.length > 0) {
        for (const line of invoice.lineItems) {
          rows.push([
            this.esc(invoice.transactionId), this.esc(invoice.type),
            this.esc(invoice.fileName), this.esc(invoice.invoiceDate),
            this.esc(invoice.supplier), this.esc(invoice.total),
            this.esc(invoice.tax), this.esc(invoice.category),
            this.esc(invoice.expenseCategory), this.esc(invoice.grossAmount),
            this.esc(invoice.netAmount), this.esc(invoice.taxAmount),
            this.esc(invoice.currency), this.esc(invoice.vendorName),
            this.esc(line.lineNumber), this.esc(line.description),
            this.esc(line.quantity), this.esc(line.netAmount),
            this.esc(line.taxAmount), this.esc(line.grossAmount),
          ].join(','));
        }
      } else {
        rows.push([
          this.esc(invoice.transactionId), this.esc(invoice.type),
          this.esc(invoice.fileName), this.esc(invoice.invoiceDate),
          this.esc(invoice.supplier), this.esc(invoice.total),
          this.esc(invoice.tax), this.esc(invoice.category),
          this.esc(invoice.expenseCategory), this.esc(invoice.grossAmount),
          this.esc(invoice.netAmount), this.esc(invoice.taxAmount),
          this.esc(invoice.currency), this.esc(invoice.vendorName),
          '', '', '', '', '', '',
        ].join(','));
      }
    }

    fs.writeFileSync(checkpointFile, rows.join('\n'), 'utf8');
    console.log(`✅ Checkpoint saved\n`);
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
    }
  }

  async run() {
    try {
      await this.init();
      await this.login();
      const invoices = await this.scrapeAllInvoices();
      this.exportToCSV(invoices);
      
      console.log('\n🎉 Done!');
      console.log(`   Total invoices: ${invoices.length}`);
      console.log(`   Output: ${CONFIG.outputFile}`);
    } catch (error) {
      console.error('\n❌ Fatal error:', error);
    } finally {
      await this.close();
    }
  }
}

const scraper = new InvoiceScraper();
scraper.run();
