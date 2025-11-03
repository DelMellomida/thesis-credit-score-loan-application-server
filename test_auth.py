#!/usr/bin/env python3
"""
Complete test script for FastAPI Auth System
Run this script to test all authentication endpoints
"""

import asyncio
import httpx
import json
from typing import Dict, Optional

# Test configuration
BASE_URL = "http://127.0.0.1:8000"  # Change this to your server URL
TEST_USER = {
    "email": "test@example.com",
    "full_name": "Test User",
    "password": "testpassword123"
}

class AuthTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        
    async def test_signup(self) -> Dict:
        """Test user registration"""
        print("\n🔐 Testing User Registration...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/signup",
                    json=TEST_USER
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 201:
                    print("✅ Registration successful!")
                    return response.json()
                else:
                    print("❌ Registration failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error during registration: {e}")
                return {"error": str(e)}
    
    async def test_login(self) -> Dict:
        """Test user login"""
        print("\n🔑 Testing User Login...")
        
        async with httpx.AsyncClient() as client:
            try:
                # OAuth2 expects form data, not JSON
                form_data = {
                    "username": TEST_USER["email"],  # OAuth2 uses 'username' field
                    "password": TEST_USER["password"]
                }
                
                response = await client.post(
                    f"{self.base_url}/auth/login",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 200:
                    print("✅ Login successful!")
                    result = response.json()
                    self.access_token = result.get("access_token")
                    return result
                else:
                    print("❌ Login failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error during login: {e}")
                return {"error": str(e)}
    
    async def test_get_current_user(self) -> Dict:
        """Test getting current user info"""
        print("\n👤 Testing Get Current User...")
        
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return {"error": "No access token"}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/auth/me",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 200:
                    print("✅ Get current user successful!")
                    return response.json()
                else:
                    print("❌ Get current user failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error getting current user: {e}")
                return {"error": str(e)}
    
    async def test_refresh_token(self) -> Dict:
        """Test token refresh"""
        print("\n🔄 Testing Token Refresh...")
        
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return {"error": "No access token"}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/refresh",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 200:
                    print("✅ Token refresh successful!")
                    result = response.json()
                    self.access_token = result.get("access_token")
                    return result
                else:
                    print("❌ Token refresh failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error refreshing token: {e}")
                return {"error": str(e)}
    
    async def test_duplicate_registration(self) -> Dict:
        """Test registering with existing email"""
        print("\n🚫 Testing Duplicate Registration...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/signup",
                    json=TEST_USER
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 400:
                    print("✅ Duplicate registration properly rejected!")
                    return response.json()
                else:
                    print("❌ Duplicate registration should have failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error during duplicate registration test: {e}")
                return {"error": str(e)}
    
    async def test_invalid_login(self) -> Dict:
        """Test login with wrong credentials"""
        print("\n❌ Testing Invalid Login...")
        
        async with httpx.AsyncClient() as client:
            try:
                form_data = {
                    "username": TEST_USER["email"],
                    "password": "wrongpassword"
                }
                
                response = await client.post(
                    f"{self.base_url}/auth/login",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 401:
                    print("✅ Invalid login properly rejected!")
                    return response.json()
                else:
                    print("❌ Invalid login should have failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error during invalid login test: {e}")
                return {"error": str(e)}
    
    async def test_unauthorized_access(self) -> Dict:
        """Test accessing protected route without token"""
        print("\n🔒 Testing Unauthorized Access...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/auth/me")
                
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
                
                if response.status_code == 401:
                    print("✅ Unauthorized access properly rejected!")
                    return response.json()
                else:
                    print("❌ Unauthorized access should have failed!")
                    return response.json()
                    
            except Exception as e:
                print(f"❌ Error during unauthorized access test: {e}")
                return {"error": str(e)}
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Auth System Tests...")
        print(f"Base URL: {self.base_url}")
        print("=" * 50)
        
        # Test 1: Registration
        await self.test_signup()
        
        # Test 2: Login
        await self.test_login()
        
        # Test 3: Get current user
        await self.test_get_current_user()
        
        # Test 4: Token refresh
        await self.test_refresh_token()
        
        # Test 5: Duplicate registration
        await self.test_duplicate_registration()
        
        # Test 6: Invalid login
        await self.test_invalid_login()
        
        # Test 7: Unauthorized access
        await self.test_unauthorized_access()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")

async def main():
    """Main test function"""
    tester = AuthTester(BASE_URL)
    await tester.run_all_tests()

if __name__ == "__main__":
    print("Auth System Tester")
    print("Make sure your FastAPI server is running on", BASE_URL)
    print("Press Ctrl+C to cancel, or Enter to continue...")
    input()
    
    asyncio.run(main())