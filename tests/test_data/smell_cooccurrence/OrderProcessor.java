package com.example.smells;

import java.util.List;
import java.util.ArrayList;
import java.util.Date;
import java.math.BigDecimal;

/**
 * Demonstrates: Long Method, Complex Method, Conditional Complexity,
 * Duplicated Code, Switch Statement, Print Statements
 *
 * Positive Dependencies:
 * - Refactoring the long processOrder() method would simultaneously solve:
 *   - Duplicated validation code
 *   - Switch statement complexity
 *   - Print statements scattered throughout
 *   - Conditional complexity from nested if-else
 */
public class OrderProcessor {

    private static final BigDecimal TAX_RATE = new BigDecimal("0.08");
    private static final BigDecimal DISCOUNT_THRESHOLD = new BigDecimal("100.00");

    // LONG METHOD + COMPLEX METHOD + DUPLICATED CODE + SWITCH STATEMENT + PRINT STATEMENTS
    public boolean processOrder(String orderId, String customerId, List<String> productIds,
                                List<Integer> quantities, List<BigDecimal> prices,
                                String paymentMethod, String shippingAddress,
                                String customerType, boolean isExpressShipping) {
        System.out.println("Processing order: " + orderId); // Print Statement smell

        // Duplicated validation #1
        if (orderId == null || orderId.trim().isEmpty()) {
            System.out.println("Error: Invalid order ID");
            return false;
        }
        if (customerId == null || customerId.trim().isEmpty()) {
            System.out.println("Error: Invalid customer ID");
            return false;
        }

        // Duplicated validation #2
        if (productIds == null || productIds.isEmpty()) {
            System.out.println("Error: No products in order");
            return false;
        }
        if (quantities == null || quantities.isEmpty()) {
            System.out.println("Error: No quantities specified");
            return false;
        }
        if (prices == null || prices.isEmpty()) {
            System.out.println("Error: No prices specified");
            return false;
        }

        // Duplicated validation #3
        if (paymentMethod == null || paymentMethod.trim().isEmpty()) {
            System.out.println("Error: Invalid payment method");
            return false;
        }
        if (shippingAddress == null || shippingAddress.trim().isEmpty()) {
            System.out.println("Error: Invalid shipping address");
            return false;
        }

        BigDecimal subtotal = BigDecimal.ZERO;
        for (int i = 0; i < productIds.size(); i++) {
            BigDecimal itemTotal = prices.get(i).multiply(new BigDecimal(quantities.get(i)));
            subtotal = subtotal.add(itemTotal);
            System.out.println("Item " + productIds.get(i) + ": " + itemTotal);
        }

        // Large SWITCH STATEMENT smell
        BigDecimal discount = BigDecimal.ZERO;
        switch (customerType) {
            case "PREMIUM":
                if (subtotal.compareTo(new BigDecimal("200")) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.15"));
                } else if (subtotal.compareTo(DISCOUNT_THRESHOLD) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.10"));
                } else {
                    discount = subtotal.multiply(new BigDecimal("0.05"));
                }
                System.out.println("Premium customer discount applied: " + discount);
                break;
            case "GOLD":
                if (subtotal.compareTo(new BigDecimal("150")) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.12"));
                } else if (subtotal.compareTo(DISCOUNT_THRESHOLD) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.08"));
                } else {
                    discount = subtotal.multiply(new BigDecimal("0.03"));
                }
                System.out.println("Gold customer discount applied: " + discount);
                break;
            case "SILVER":
                if (subtotal.compareTo(DISCOUNT_THRESHOLD) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.05"));
                }
                System.out.println("Silver customer discount applied: " + discount);
                break;
            case "REGULAR":
                if (subtotal.compareTo(new BigDecimal("200")) >= 0) {
                    discount = subtotal.multiply(new BigDecimal("0.02"));
                }
                System.out.println("Regular customer discount: " + discount);
                break;
            default:
                System.out.println("Unknown customer type, no discount");
                break;
        }

        BigDecimal discountedTotal = subtotal.subtract(discount);
        BigDecimal tax = discountedTotal.multiply(TAX_RATE);

        // More conditional complexity
        BigDecimal shippingCost = BigDecimal.ZERO;
        if (isExpressShipping) {
            if (discountedTotal.compareTo(new BigDecimal("50")) < 0) {
                shippingCost = new BigDecimal("15.00");
            } else if (discountedTotal.compareTo(DISCOUNT_THRESHOLD) < 0) {
                shippingCost = new BigDecimal("10.00");
            } else {
                shippingCost = new BigDecimal("5.00");
            }
        } else {
            if (discountedTotal.compareTo(new BigDecimal("50")) < 0) {
                shippingCost = new BigDecimal("8.00");
            } else if (discountedTotal.compareTo(DISCOUNT_THRESHOLD) < 0) {
                shippingCost = new BigDecimal("5.00");
            } else {
                shippingCost = BigDecimal.ZERO;
            }
        }

        BigDecimal finalTotal = discountedTotal.add(tax).add(shippingCost);

        System.out.println("Subtotal: " + subtotal);
        System.out.println("Discount: " + discount);
        System.out.println("Tax: " + tax);
        System.out.println("Shipping: " + shippingCost);
        System.out.println("Final Total: " + finalTotal);

        // Another large switch for payment processing
        boolean paymentSuccessful = false;
        switch (paymentMethod) {
            case "CREDIT_CARD":
                System.out.println("Processing credit card payment");
                paymentSuccessful = true;
                break;
            case "DEBIT_CARD":
                System.out.println("Processing debit card payment");
                paymentSuccessful = true;
                break;
            case "PAYPAL":
                System.out.println("Processing PayPal payment");
                paymentSuccessful = true;
                break;
            case "BANK_TRANSFER":
                System.out.println("Processing bank transfer");
                paymentSuccessful = true;
                break;
            default:
                System.out.println("Unsupported payment method: " + paymentMethod);
                paymentSuccessful = false;
        }

        if (!paymentSuccessful) {
            System.out.println("Payment failed for order: " + orderId);
            return false;
        }

        System.out.println("Order " + orderId + " processed successfully!");
        return true;
    }
}
