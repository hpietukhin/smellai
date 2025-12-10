package com.example.smells;

import java.math.BigDecimal;
import java.util.Date;

/**
 * Demonstrates: Duplicated Conditions, Duplicated Code, Complex Method
 *
 * DUPLICATED CONDITIONS - Same conditional logic repeated in multiple places
 *
 * Positive Dependencies:
 * - Fixing Duplicated Conditions would also solve:
 *   - Duplicated Code (the repeated validation blocks)
 *   - Reduce overall method complexity
 */
public class PaymentValidator {

    private static final BigDecimal MAX_TRANSACTION_AMOUNT = new BigDecimal("10000.00");
    private static final BigDecimal MIN_TRANSACTION_AMOUNT = new BigDecimal("0.01");

    // DUPLICATED CONDITIONS #1
    public boolean validateCreditCardPayment(String cardNumber, String cvv, Date expiryDate,
                                            BigDecimal amount, String cardHolder) {
        // Duplicated validation block
        if (cardNumber == null || cardNumber.trim().isEmpty()) {
            return false;
        }
        if (cvv == null || cvv.trim().isEmpty()) {
            return false;
        }
        if (expiryDate == null || expiryDate.before(new Date())) {
            return false;
        }
        if (amount == null || amount.compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
            return false;
        }
        if (amount.compareTo(MAX_TRANSACTION_AMOUNT) > 0) {
            return false;
        }
        if (cardHolder == null || cardHolder.trim().isEmpty()) {
            return false;
        }

        return cardNumber.length() >= 13 && cardNumber.length() <= 19 &&
               cvv.length() == 3 || cvv.length() == 4;
    }

    // DUPLICATED CONDITIONS #2 - Almost identical to above
    public boolean validateDebitCardPayment(String cardNumber, String pin, Date expiryDate,
                                           BigDecimal amount, String cardHolder) {
        // Same validation block duplicated
        if (cardNumber == null || cardNumber.trim().isEmpty()) {
            return false;
        }
        if (pin == null || pin.trim().isEmpty()) {
            return false;
        }
        if (expiryDate == null || expiryDate.before(new Date())) {
            return false;
        }
        if (amount == null || amount.compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
            return false;
        }
        if (amount.compareTo(MAX_TRANSACTION_AMOUNT) > 0) {
            return false;
        }
        if (cardHolder == null || cardHolder.trim().isEmpty()) {
            return false;
        }

        return cardNumber.length() >= 13 && cardNumber.length() <= 19 &&
               pin.length() == 4;
    }

    // DUPLICATED CONDITIONS #3 - Yet another copy
    public boolean validatePayPalPayment(String email, String password, BigDecimal amount) {
        // Partially duplicated validation
        if (email == null || email.trim().isEmpty()) {
            return false;
        }
        if (password == null || password.trim().isEmpty()) {
            return false;
        }
        if (amount == null || amount.compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
            return false;
        }
        if (amount.compareTo(MAX_TRANSACTION_AMOUNT) > 0) {
            return false;
        }

        return email.contains("@") && email.contains(".");
    }

    // DUPLICATED CONDITIONS #4
    public boolean validateBankTransfer(String accountNumber, String routingNumber,
                                       BigDecimal amount, String accountHolder) {
        // Same validation patterns again
        if (accountNumber == null || accountNumber.trim().isEmpty()) {
            return false;
        }
        if (routingNumber == null || routingNumber.trim().isEmpty()) {
            return false;
        }
        if (amount == null || amount.compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
            return false;
        }
        if (amount.compareTo(MAX_TRANSACTION_AMOUNT) > 0) {
            return false;
        }
        if (accountHolder == null || accountHolder.trim().isEmpty()) {
            return false;
        }

        return accountNumber.length() >= 8 && accountNumber.length() <= 17 &&
               routingNumber.length() == 9;
    }

    // DUPLICATED CONDITIONS in business logic
    public String determinePaymentRisk(BigDecimal amount, String country, boolean isNewCustomer,
                                      int previousTransactions) {
        // Risk assessment with duplicated conditional logic
        if (amount.compareTo(new BigDecimal("1000")) > 0) {
            if (isNewCustomer) {
                if (country.equals("US") || country.equals("CA") || country.equals("UK")) {
                    return "MEDIUM_RISK";
                } else {
                    return "HIGH_RISK";
                }
            } else {
                if (previousTransactions > 10) {
                    return "LOW_RISK";
                } else {
                    return "MEDIUM_RISK";
                }
            }
        } else if (amount.compareTo(new BigDecimal("100")) > 0) {
            if (isNewCustomer) {
                if (country.equals("US") || country.equals("CA") || country.equals("UK")) {
                    return "LOW_RISK";
                } else {
                    return "MEDIUM_RISK";
                }
            } else {
                return "LOW_RISK";
            }
        } else {
            return "LOW_RISK";
        }
    }

    // DUPLICATED CONDITIONS - Similar risk logic repeated
    public boolean requiresAdditionalVerification(BigDecimal amount, String country,
                                                 boolean isNewCustomer, int previousTransactions) {
        // Almost identical logic to determinePaymentRisk
        if (amount.compareTo(new BigDecimal("1000")) > 0) {
            if (isNewCustomer) {
                if (country.equals("US") || country.equals("CA") || country.equals("UK")) {
                    return true;
                } else {
                    return true;
                }
            } else {
                if (previousTransactions > 10) {
                    return false;
                } else {
                    return true;
                }
            }
        } else if (amount.compareTo(new BigDecimal("100")) > 0) {
            if (isNewCustomer) {
                if (country.equals("US") || country.equals("CA") || country.equals("UK")) {
                    return false;
                } else {
                    return true;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    // DUPLICATED CONDITIONS - Same null/empty checks everywhere
    public boolean validateRefundRequest(String transactionId, BigDecimal refundAmount,
                                        String reason, Date transactionDate) {
        // Yet another set of duplicated validations
        if (transactionId == null || transactionId.trim().isEmpty()) {
            return false;
        }
        if (refundAmount == null || refundAmount.compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
            return false;
        }
        if (reason == null || reason.trim().isEmpty()) {
            return false;
        }
        if (transactionDate == null || transactionDate.after(new Date())) {
            return false;
        }

        // Check if refund is within 30 days
        long daysSinceTransaction = (new Date().getTime() - transactionDate.getTime()) / (1000 * 60 * 60 * 24);
        return daysSinceTransaction <= 30;
    }
}
