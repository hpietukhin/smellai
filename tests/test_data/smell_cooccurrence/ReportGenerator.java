package com.example.smells;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.math.BigDecimal;

/**
 * Demonstrates: God Class/Large Class, Data Clumps, Feature Envy, Long Method, Print Statements
 *
 * GOD CLASS characteristics:
 * - Over 200 lines
 * - Multiple unrelated responsibilities (reporting, formatting, calculation, data access)
 * - Too many methods and fields
 * - High coupling to many other classes
 *
 * Positive Dependencies:
 * - Refactoring Large Class would solve:
 *   - Data Clumps (same parameters appear together repeatedly)
 *   - Feature Envy (methods working more with external data)
 *   - Long Methods (distributed across extracted classes)
 */
public class ReportGenerator {

    // Too many fields for different responsibilities
    private DatabaseConnection dbConnection;
    private List<SalesRecord> salesData;
    private List<CustomerRecord> customerData;
    private List<InventoryRecord> inventoryData;
    private Map<String, BigDecimal> revenueByProduct;
    private Map<String, Integer> salesByRegion;
    private SimpleDateFormat dateFormat;
    private String currentUser;
    private Date reportDate;
    private String reportType;
    private boolean includeCharts;
    private boolean includeDetails;

    // DATA CLUMPS: These 4 parameters always appear together
    // startDate, endDate, region, category appear in multiple methods
    public String generateSalesReport(Date startDate, Date endDate, String region, String category) {
        System.out.println("Generating sales report..."); // Print Statement

        // Feature Envy - working more with dbConnection than own data
        salesData = dbConnection.fetchSalesData(startDate, endDate, region, category);

        StringBuilder report = new StringBuilder();
        report.append(formatReportHeader(startDate, endDate, region, category)); // DATA CLUMP
        report.append(calculateTotalSales(startDate, endDate, region, category)); // DATA CLUMP
        report.append(generateSalesSummary(startDate, endDate, region, category)); // DATA CLUMP

        System.out.println("Sales report generated"); // Print Statement
        return report.toString();
    }

    // LONG METHOD + DATA CLUMPS
    public String generateCustomerReport(Date startDate, Date endDate, String region, String category) {
        System.out.println("Generating customer report...");

        // Feature Envy - mostly working with dbConnection
        customerData = dbConnection.fetchCustomerData(startDate, endDate, region, category);

        StringBuilder report = new StringBuilder();
        report.append("Customer Report\n");
        report.append("Period: ").append(dateFormat.format(startDate))
              .append(" to ").append(dateFormat.format(endDate)).append("\n");
        report.append("Region: ").append(region).append("\n");
        report.append("Category: ").append(category).append("\n\n");

        int totalCustomers = 0;
        int newCustomers = 0;
        int returningCustomers = 0;
        BigDecimal totalRevenue = BigDecimal.ZERO;
        BigDecimal avgOrderValue = BigDecimal.ZERO;

        // Feature Envy - iterating over external data structure
        for (CustomerRecord customer : customerData) {
            totalCustomers++;
            if (customer.getFirstPurchaseDate().after(startDate)) {
                newCustomers++;
            } else {
                returningCustomers++;
            }
            totalRevenue = totalRevenue.add(customer.getTotalSpent());
        }

        if (totalCustomers > 0) {
            avgOrderValue = totalRevenue.divide(new BigDecimal(totalCustomers), 2, BigDecimal.ROUND_HALF_UP);
        }

        report.append("Total Customers: ").append(totalCustomers).append("\n");
        report.append("New Customers: ").append(newCustomers).append("\n");
        report.append("Returning Customers: ").append(returningCustomers).append("\n");
        report.append("Total Revenue: $").append(totalRevenue).append("\n");
        report.append("Average Order Value: $").append(avgOrderValue).append("\n");

        System.out.println("Customer report completed");
        return report.toString();
    }

    // DATA CLUMPS - same 4 parameters again
    public String generateInventoryReport(Date startDate, Date endDate, String region, String category) {
        System.out.println("Generating inventory report...");

        // Feature Envy
        inventoryData = dbConnection.fetchInventoryData(startDate, endDate, region, category);

        StringBuilder report = new StringBuilder();
        report.append(formatReportHeader(startDate, endDate, region, category));

        int totalItems = 0;
        int lowStockItems = 0;
        int outOfStockItems = 0;

        // Feature Envy - working with external collection
        for (InventoryRecord item : inventoryData) {
            totalItems++;
            if (item.getQuantity() == 0) {
                outOfStockItems++;
            } else if (item.getQuantity() < item.getReorderLevel()) {
                lowStockItems++;
            }
        }

        report.append("Total Items: ").append(totalItems).append("\n");
        report.append("Low Stock: ").append(lowStockItems).append("\n");
        report.append("Out of Stock: ").append(outOfStockItems).append("\n");

        System.out.println("Inventory report completed");
        return report.toString();
    }

    // DATA CLUMPS - same parameters
    private String formatReportHeader(Date startDate, Date endDate, String region, String category) {
        StringBuilder header = new StringBuilder();
        header.append("=".repeat(50)).append("\n");
        header.append("Report Generated: ").append(dateFormat.format(new Date())).append("\n");
        header.append("Period: ").append(dateFormat.format(startDate))
              .append(" to ").append(dateFormat.format(endDate)).append("\n");
        header.append("Region: ").append(region).append("\n");
        header.append("Category: ").append(category).append("\n");
        header.append("=".repeat(50)).append("\n\n");
        return header.toString();
    }

    // DATA CLUMPS + LONG METHOD
    private String calculateTotalSales(Date startDate, Date endDate, String region, String category) {
        BigDecimal totalRevenue = BigDecimal.ZERO;
        int totalOrders = 0;
        Map<String, BigDecimal> productRevenue = new HashMap<>();

        // Feature Envy - working with salesData collection
        for (SalesRecord sale : salesData) {
            totalOrders++;
            totalRevenue = totalRevenue.add(sale.getAmount());

            String productId = sale.getProductId();
            productRevenue.put(productId,
                productRevenue.getOrDefault(productId, BigDecimal.ZERO).add(sale.getAmount()));
        }

        StringBuilder result = new StringBuilder();
        result.append("Total Orders: ").append(totalOrders).append("\n");
        result.append("Total Revenue: $").append(totalRevenue).append("\n");
        result.append("\nTop Products:\n");

        // Feature Envy - sorting external data
        productRevenue.entrySet().stream()
            .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
            .limit(10)
            .forEach(entry -> result.append("  ")
                .append(entry.getKey())
                .append(": $")
                .append(entry.getValue())
                .append("\n"));

        return result.toString();
    }

    // DATA CLUMPS
    private String generateSalesSummary(Date startDate, Date endDate, String region, String category) {
        // Feature Envy - working with salesData
        Map<String, Integer> salesByDay = new HashMap<>();

        for (SalesRecord sale : salesData) {
            String day = dateFormat.format(sale.getDate());
            salesByDay.put(day, salesByDay.getOrDefault(day, 0) + 1);
        }

        StringBuilder summary = new StringBuilder("\nDaily Sales Summary:\n");
        salesByDay.forEach((day, count) ->
            summary.append("  ").append(day).append(": ").append(count).append(" sales\n"));

        return summary.toString();
    }

    // More methods showing Feature Envy
    public void exportToCSV(String filename, Date startDate, Date endDate, String region, String category) {
        System.out.println("Exporting to CSV: " + filename);
        // Feature Envy - working with external FileWriter
        FileWriter writer = new FileWriter(filename);
        writer.write(generateSalesReport(startDate, endDate, region, category));
        writer.close();
    }

    public void exportToPDF(String filename, Date startDate, Date endDate, String region, String category) {
        System.out.println("Exporting to PDF: " + filename);
        // Feature Envy - working with external PDFGenerator
        PDFGenerator pdf = new PDFGenerator();
        pdf.addContent(generateSalesReport(startDate, endDate, region, category));
        pdf.save(filename);
    }

    public void sendReportByEmail(String recipient, Date startDate, Date endDate,
                                 String region, String category) {
        System.out.println("Sending report to: " + recipient);
        // Feature Envy - working with external EmailService
        EmailService email = new EmailService();
        String report = generateSalesReport(startDate, endDate, region, category);
        email.send(recipient, "Sales Report", report);
    }
}

// Stub classes for compilation
class SalesRecord {
    private Date date;
    private String productId;
    private BigDecimal amount;

    public Date getDate() { return date; }
    public String getProductId() { return productId; }
    public BigDecimal getAmount() { return amount; }
}

class CustomerRecord {
    private Date firstPurchaseDate;
    private BigDecimal totalSpent;

    public Date getFirstPurchaseDate() { return firstPurchaseDate; }
    public BigDecimal getTotalSpent() { return totalSpent; }
}

class InventoryRecord {
    private int quantity;
    private int reorderLevel;

    public int getQuantity() { return quantity; }
    public int getReorderLevel() { return reorderLevel; }
}

class DatabaseConnection {
    public List<SalesRecord> fetchSalesData(Date start, Date end, String region, String category) {
        return new ArrayList<>();
    }
    public List<CustomerRecord> fetchCustomerData(Date start, Date end, String region, String category) {
        return new ArrayList<>();
    }
    public List<InventoryRecord> fetchInventoryData(Date start, Date end, String region, String category) {
        return new ArrayList<>();
    }
}

class FileWriter {
    public FileWriter(String filename) {}
    public void write(String content) {}
    public void close() {}
}

class PDFGenerator {
    public void addContent(String content) {}
    public void save(String filename) {}
}

class EmailService {
    public void send(String recipient, String subject, String body) {}
}
