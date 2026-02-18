#!/usr/bin/env python3
"""
Backend API Testing Suite for ETL-ML Platform
Tests all API endpoints with comprehensive coverage
"""

import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

class ETLMLAPITester:
    def __init__(self, base_url: str = "https://data-engine-labs.preview.emergentagent.com"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        
    def log_result(self, test_name: str, success: bool, response_data: Any = None, error: str = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {test_name} - PASSED")
        else:
            self.failed_tests.append({"test": test_name, "error": error})
            print(f"❌ {test_name} - FAILED: {error}")
        
        if response_data and isinstance(response_data, dict):
            if 'error' in str(response_data).lower():
                print(f"   Response: {response_data}")

    def test_api_call(self, method: str, endpoint: str, expected_status: int = 200, 
                     data: Dict = None, test_name: str = None) -> tuple:
        """Generic API test method"""
        if not test_name:
            test_name = f"{method} {endpoint}"
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=15)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check status code
            success = response.status_code == expected_status
            
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            if not success:
                error_msg = f"Expected status {expected_status}, got {response.status_code}"
                if response_data:
                    error_msg += f" - Response: {response_data}"
                self.log_result(test_name, False, error=error_msg)
                return False, None
            
            self.log_result(test_name, True, response_data)
            return True, response_data
            
        except requests.exceptions.RequestException as e:
            self.log_result(test_name, False, error=f"Network error: {str(e)}")
            return False, None
        except Exception as e:
            self.log_result(test_name, False, error=f"Unexpected error: {str(e)}")
            return False, None

    def test_health_endpoint(self):
        """Test API health check"""
        print("\n🔍 Testing Health Endpoint...")
        success, data = self.test_api_call("GET", "/api/health", test_name="API Health Check")
        
        if success and data:
            if isinstance(data, dict) and data.get('status') == 'healthy':
                print(f"   ✓ API is healthy, version: {data.get('version', 'unknown')}")
                print(f"   ✓ Database: {data.get('database', 'unknown')}")
                print(f"   ✓ WebSocket connections: {data.get('websocket_connections', 0)}")
                return True
        return False

    def test_dashboard_endpoints(self):
        """Test dashboard-related endpoints"""
        print("\n🔍 Testing Dashboard Endpoints...")
        
        # Test dashboard stats
        success1, stats_data = self.test_api_call("GET", "/api/dashboard/stats", test_name="Dashboard Stats")
        
        # Test dashboard metrics
        success2, metrics_data = self.test_api_call("GET", "/api/dashboard/metrics", test_name="Dashboard Metrics")
        
        # Test recent runs
        success3, runs_data = self.test_api_call("GET", "/api/dashboard/recent-runs?limit=5", test_name="Recent Runs")
        
        if success1 and stats_data:
            print(f"   ✓ Stats - Pipelines: {stats_data.get('total_pipelines', 0)}, Experiments: {stats_data.get('total_experiments', 0)}")
            
        return success1 and success2 and success3

    def test_pipeline_endpoints(self):
        """Test pipeline management endpoints"""
        print("\n🔍 Testing Pipeline Endpoints...")
        
        # List pipelines
        success1, pipelines_data = self.test_api_call("GET", "/api/pipelines", test_name="List Pipelines")
        
        if not success1:
            return False
            
        pipeline_id = None
        if pipelines_data and len(pipelines_data) > 0:
            pipeline_id = pipelines_data[0].get('id')
            print(f"   ✓ Found {len(pipelines_data)} pipelines")
            
            # Get specific pipeline
            if pipeline_id:
                success2, pipeline_data = self.test_api_call("GET", f"/api/pipelines/{pipeline_id}", test_name="Get Pipeline Details")
                
                # Test run pipeline
                success3, run_data = self.test_api_call("POST", f"/api/pipelines/{pipeline_id}/run", test_name="Run Pipeline")
                
                if success3 and run_data:
                    run_id = run_data.get('id')
                    print(f"   ✓ Pipeline run started with ID: {run_id}")
                    
                # Get pipeline runs
                success4, runs_data = self.test_api_call("GET", f"/api/pipelines/{pipeline_id}/runs", test_name="Get Pipeline Runs")
                
                return success2 and success3 and success4
        else:
            print("   ⚠️  No pipelines found - this might be expected for a fresh installation")
            return True
            
        return success1

    def test_experiment_endpoints(self):
        """Test experiment management endpoints"""
        print("\n🔍 Testing Experiment Endpoints...")
        
        # List experiments
        success1, experiments_data = self.test_api_call("GET", "/api/experiments", test_name="List Experiments")
        
        if success1:
            print(f"   ✓ Found {len(experiments_data) if experiments_data else 0} experiments")
            
            # Create a new experiment
            experiment_payload = {
                "name": f"Test Experiment {datetime.now().strftime('%H%M%S')}",
                "description": "Automated test experiment",
                "algorithm": "RandomForest",
                "parameters": {
                    "n_estimators": 50,
                    "max_depth": 10
                }
            }
            
            success2, exp_data = self.test_api_call("POST", "/api/experiments", 200, 
                                                  experiment_payload, "Create Experiment")
            
            if success2 and exp_data:
                exp_id = exp_data.get('id')
                print(f"   ✓ Created experiment with ID: {exp_id}")
                
                # Get experiment details
                success3, details = self.test_api_call("GET", f"/api/experiments/{exp_id}", 
                                                     test_name="Get Experiment Details")
                
                return success3
        
        return success1

    def test_automl_endpoints(self):
        """Test AutoML endpoints"""
        print("\n🔍 Testing AutoML Endpoints...")
        
        # List AutoML runs
        success1, automl_data = self.test_api_call("GET", "/api/automl/runs", test_name="List AutoML Runs")
        
        if success1:
            print(f"   ✓ Found {len(automl_data) if automl_data else 0} AutoML runs")
            
            # Create AutoML run
            automl_payload = {
                "experiment_name": f"Test AutoML {datetime.now().strftime('%H%M%S')}",
                "description": "Automated test AutoML run",
                "algorithms": ["RandomForest", "LogisticRegression"],
                "cv_folds": 3,
                "max_trials": 5
            }
            
            success2, run_data = self.test_api_call("POST", "/api/automl/run", 200,
                                                  automl_payload, "Start AutoML Run")
            
            if success2 and run_data:
                run_id = run_data.get('id')
                print(f"   ✓ Started AutoML run with ID: {run_id}")
                
                # Get AutoML run details
                success3, details = self.test_api_call("GET", f"/api/automl/runs/{run_id}",
                                                     test_name="Get AutoML Run Details")
                return success3
        
        return success1

    def test_validation_endpoints(self):
        """Test data validation endpoints"""
        print("\n🔍 Testing Validation Endpoints...")
        
        # List validations
        success1, validations_data = self.test_api_call("GET", "/api/validations", test_name="List Validations")
        
        if success1:
            print(f"   ✓ Found {len(validations_data) if validations_data else 0} validations")
            
            # Create validation
            validation_payload = {
                "name": f"Test Validation {datetime.now().strftime('%H%M%S')}",
                "dataset_path": "/test/data/sample.csv",
                "rules": [
                    {"rule": "not_null", "column": "id"},
                    {"rule": "type_check", "column": "value", "type": "numeric"}
                ]
            }
            
            success2, val_data = self.test_api_call("POST", "/api/validations", 200,
                                                  validation_payload, "Create Validation")
            
            if success2 and val_data:
                val_id = val_data.get('id')
                print(f"   ✓ Created validation with ID: {val_id}")
        
        return success1

    def test_model_endpoints(self):
        """Test model management endpoints"""
        print("\n🔍 Testing Model Endpoints...")
        
        # List models
        success1, models_data = self.test_api_call("GET", "/api/models", test_name="List Models")
        
        if success1:
            print(f"   ✓ Found {len(models_data) if models_data else 0} models")
        
        return success1

    def test_log_endpoints(self):
        """Test logging endpoints"""
        print("\n🔍 Testing Log Endpoints...")
        
        # List logs
        success1, logs_data = self.test_api_call("GET", "/api/logs?limit=10", test_name="List Logs")
        
        if success1:
            print(f"   ✓ Found {len(logs_data) if logs_data else 0} log entries")
            
            # Test filtered logs
            success2, error_logs = self.test_api_call("GET", "/api/logs?level=ERROR&limit=5", 
                                                    test_name="Filter Error Logs")
            return success2
        
        return success1

    def test_seed_endpoint(self):
        """Test database seeding"""
        print("\n🔍 Testing Seed Endpoint...")
        
        success, seed_data = self.test_api_call("POST", "/api/seed", test_name="Seed Database")
        
        if success:
            print("   ✓ Database seeded successfully")
        
        return success

    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting ETL-ML Platform API Testing Suite")
        print("=" * 60)
        
        # Test in logical order
        test_methods = [
            self.test_health_endpoint,
            self.test_dashboard_endpoints,
            self.test_pipeline_endpoints,
            self.test_experiment_endpoints,
            self.test_automl_endpoints,
            self.test_validation_endpoints,
            self.test_model_endpoints,
            self.test_log_endpoints,
            self.test_seed_endpoint,
        ]
        
        start_time = datetime.now()
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ {test_method.__name__} failed with exception: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0.0%")
        print(f"Duration: {duration:.2f} seconds")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for i, failure in enumerate(self.failed_tests, 1):
                print(f"{i}. {failure['test']}: {failure['error']}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test runner"""
    tester = ETLMLAPITester()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())