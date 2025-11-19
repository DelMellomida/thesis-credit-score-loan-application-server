"""
SMS Service using PhilSMS API for sending transactional messages.
PhilSMS provides 5 free SMS credits to test their service.
Pricing: ₱0.35 per SMS with no minimum top-up amount.
"""

import logging
from typing import Optional
import httpx
from app.core import Settings

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending SMS messages via PhilSMS API."""
    
    def __init__(self):
        """Initialize PhilSMS client with credentials from environment variables."""
        self.api_token = Settings.PHILSMS_API_TOKEN  # Get from PhilSMS Dashboard
        self.sender_id = Settings.PHILSMS_SENDER_ID  # Your brand name (max 11 chars)
        self.api_url = "https://dashboard.philsms.com/api/v3/sms/send"
        
        if self.api_token and self.sender_id:
            logger.info("PhilSMS service initialized successfully")
        else:
            logger.warning(
                "PhilSMS credentials not configured. Set PHILSMS_API_TOKEN "
                "and PHILSMS_SENDER_ID environment variables"
            )
    
    @staticmethod
    def _sanitize_message(message: str) -> str:
        """
        Remove or replace special characters that might trigger unicode encoding.
        
        Args:
            message: Original message text
            
        Returns:
            str: Sanitized message with only plain ASCII characters
        """
        # Replace common special characters
        replacements = {
            '₱': 'PHP ',
            'é': 'e',
            'ñ': 'n',
            'Ñ': 'N',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '–': '-',
            '—': '-',
        }
        
        for old, new in replacements.items():
            message = message.replace(old, new)
        
        # Keep only ASCII printable characters
        message = ''.join(char if ord(char) < 128 else '' for char in message)
        
        return message
    
    async def send_sms(
        self,
        phone_number: str,
        message: str,
        sender_id: Optional[str] = None
    ) -> dict:
        """
        Send an SMS message via PhilSMS.
        
        Args:
            phone_number: Recipient's phone number (with or without +63 for Philippines)
            message: SMS message content
            sender_id: Optional custom sender ID (default: uses configured sender_id)
            
        Returns:
            dict: Response containing success status and message details
            
        Raises:
            Exception: If PhilSMS is not configured or request fails
        """
        if not self.api_token or not self.sender_id:
            raise Exception(
                "PhilSMS service not initialized. Please configure "
                "PHILSMS_API_TOKEN and PHILSMS_SENDER_ID"
            )
        
        try:
            # Normalize phone number
            normalized_number = self._normalize_phone_number(phone_number)
            
            # Sanitize message to ensure plain text
            sanitized_message = self._sanitize_message(message)
            
            # Use custom sender_id if provided, otherwise use default
            active_sender_id = sender_id if sender_id else self.sender_id
            
            # Log the SMS attempt (without sensitive details)
            logger.info(
                f"Attempting to send SMS via PhilSMS to {normalized_number[:5]}...***. "
                f"Message length: {len(sanitized_message)} chars, Sender: {active_sender_id}"
            )
            
            # Prepare request headers and payload
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "recipient": normalized_number,
                "sender_id": active_sender_id,
                "type": "plain",
                "message": sanitized_message
            }
            
            # Send SMS via PhilSMS API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                # Parse response
                try:
                    response_data = response.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse JSON response: {response.text}")
                    raise Exception(f"Invalid JSON response from PhilSMS: {response.text}")
                
                # Check for success based on PhilSMS actual response format
                # PhilSMS returns: {"status": "success", "message": "...", "data": {...}}
                if response_data.get("status") == "success":
                    # Extract data from the nested "data" object
                    data = response_data.get("data", {})
                    
                    result = {
                        "success": True,
                        "message_id": data.get("uid"),  # PhilSMS uses "uid" not "id"
                        "status": data.get("status"),  # Delivery status (e.g., "Delivered")
                        "phone": normalized_number,
                        "sender": active_sender_id,
                        "cost": data.get("cost"),
                        "sms_count": data.get("sms_count")
                    }
                    
                    logger.info(
                        f"SMS sent successfully via PhilSMS to {normalized_number[:5]}...***. "
                        f"UID: {data.get('uid')}, Status: {data.get('status')}, Cost: PHP{data.get('cost')}"
                    )
                    
                    return result
                else:
                    # Handle API error
                    # PhilSMS error response format: {"status": "error", "message": "..."}
                    error_message = response_data.get("message", "Unknown error")
                    error_status = response_data.get("status", "failed")
                    
                    logger.error(
                        f"PhilSMS API error: HTTP {response.status_code} - "
                        f"Status: {error_status}, Message: {error_message}"
                    )
                    
                    raise Exception(f"PhilSMS API error: {error_message}")
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error sending SMS via PhilSMS: {str(e)}")
            raise Exception(f"Failed to send SMS: {str(e)}")
        except Exception as e:
            logger.error(f"PhilSMS error sending SMS: {str(e)}")
            raise
    
    @staticmethod
    def _normalize_phone_number(phone_number: str) -> str:
        """
        Normalize phone number to Philippine format (639XXXXXXXXX).
        PhilSMS accepts format without the + prefix.
        
        Args:
            phone_number: Phone number in various formats
                         Examples: "9123456789", "09123456789", "+639123456789", "639123456789"
        
        Returns:
            str: Normalized phone number in format "639123456789"
            
        Raises:
            ValueError: If phone number format is invalid
        """
        if not phone_number:
            raise ValueError("Phone number cannot be empty")
        
        # Remove spaces, hyphens, and other common separators
        cleaned = phone_number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        # Remove + if present
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        
        # Remove leading zeros if present (PH numbers)
        if cleaned.startswith("0"):
            cleaned = cleaned[1:]
        
        # Add country code 63 if not present
        if not cleaned.startswith("63"):
            cleaned = f"63{cleaned}"
        
        # Validate length (63 + 10 digits = 12 total)
        if len(cleaned) != 12:
            raise ValueError(
                f"Invalid phone number format. Expected 12 digits (63XXXXXXXXXX), got {len(cleaned)}"
            )
        
        # Validate all characters are digits
        if not cleaned.isdigit():
            raise ValueError("Phone number must contain only digits")
        
        # PhilSMS expects format without + prefix
        return cleaned
    
    async def send_loan_approval_sms(
        self,
        phone_number: str,
        applicant_name: str,
        loan_amount: str
    ) -> dict:
        """
        Send loan approval notification SMS.
        
        Args:
            phone_number: Recipient's phone number
            applicant_name: Name of the applicant
            loan_amount: Approved loan amount
            
        Returns:
            dict: Response from PhilSMS API
        """
        message = (
            f"Hello {applicant_name}! Your loan application with a max loan of PHP {loan_amount} "
            f"has been APPROVED. Please visit our branch as soon as possible to "
            f"complete the signing and processing. Thank you!"
        )
        
        return await self.send_sms(phone_number, message)
    
    async def send_loan_denial_sms(
        self,
        phone_number: str,
        applicant_name: str
    ) -> dict:
        """
        Send loan denial notification SMS.
        
        Args:
            phone_number: Recipient's phone number
            applicant_name: Name of the applicant
            
        Returns:
            dict: Response from PhilSMS API
        """
        message = (
            f"Hello {applicant_name}, your loan application has been reviewed. "
            f"Unfortunately, it does not meet our current approval criteria. "
            f"Please contact our office for more information. Thank you."
        )
        
        return await self.send_sms(phone_number, message)
    
    async def send_custom_sms(
        self,
        phone_number: str,
        message: str,
        message_type: str = "general"
    ) -> dict:
        """
        Send a custom SMS message.
        
        Args:
            phone_number: Recipient's phone number
            message: Custom message text
            message_type: Type of message for logging (e.g., "general", "alert", "reminder")
            
        Returns:
            dict: Response from PhilSMS API
        """
        logger.info(f"Sending {message_type} SMS to {phone_number[:5]}...***")
        return await self.send_sms(phone_number, message)
    
    async def check_balance(self) -> dict:
        """
        Check SMS balance/credits in PhilSMS account.
        Note: This requires the profile API endpoint if available.
        
        Returns:
            dict: Balance information
        """
        if not self.api_token:
            raise Exception("PhilSMS API token not configured")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/json"
            }
            
            # PhilSMS profile endpoint (check documentation for exact URL)
            balance_url = "https://app.philsms.com/api/v3/profile"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(balance_url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"PhilSMS balance checked successfully")
                    return {
                        "success": True,
                        "balance": data.get("balance", "N/A"),
                        "data": data
                    }
                else:
                    logger.error(f"Failed to check balance: {response.status_code}")
                    return {
                        "success": False,
                        "error": "Failed to retrieve balance"
                    }
        except Exception as e:
            logger.error(f"Error checking PhilSMS balance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
notification_service = NotificationService()