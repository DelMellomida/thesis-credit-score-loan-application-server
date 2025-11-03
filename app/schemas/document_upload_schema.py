from fastapi import UploadFile, File, Form
from typing import Optional

class DocumentUploadRequest:
    def __init__(
        self,
        brgyCert: Optional[UploadFile] = File(None),
        eSignaturePersonal: Optional[UploadFile] = File(None),
        payslip: Optional[UploadFile] = File(None),
        companyId: Optional[UploadFile] = File(None),
        proofOfBilling: Optional[UploadFile] = File(None),
        eSignatureCoMaker: Optional[UploadFile] = File(None),
    ):
        self.brgyCert = brgyCert
        self.eSignaturePersonal = eSignaturePersonal
        self.payslip = payslip
        self.companyId = companyId
        self.proofOfBilling = proofOfBilling
        self.eSignatureCoMaker = eSignatureCoMaker

    def to_dict(self):
        return {
            "brgy_cert": self.brgyCert,
            "e_signature_personal": self.eSignaturePersonal,
            "payslip": self.payslip,
            "company_id": self.companyId,
            "proof_of_billing": self.proofOfBilling,
            "e_signature_comaker": self.eSignatureCoMaker,
        }
