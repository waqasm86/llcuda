"""
Test suite for llcuda Python package
"""

import pytest
import sys
import os

# Add parent directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import():
    """Test that the package can be imported"""
    import llcuda
    assert llcuda.__version__ == "0.1.0"


def test_cuda_check():
    """Test CUDA availability check"""
    import llcuda
    # This will be False in non-GPU environments, which is expected
    result = llcuda.check_cuda_available()
    assert isinstance(result, bool)


def test_inference_engine_creation():
    """Test creating an InferenceEngine instance"""
    import llcuda
    engine = llcuda.InferenceEngine()
    assert engine is not None
    assert hasattr(engine, 'load_model')
    assert hasattr(engine, 'infer')
    assert hasattr(engine, 'get_metrics')


def test_infer_request_creation():
    """Test creating InferRequest objects"""
    try:
        from llcuda import _llcuda
        request = _llcuda.InferRequest()
        assert request is not None
        
        # Test setting properties
        request.prompt = "Test prompt"
        request.max_tokens = 100
        request.temperature = 0.7
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert abs(request.temperature - 0.7) < 0.01
    except ImportError:
        pytest.skip("C++ extension not built")


def test_model_config():
    """Test ModelConfig creation"""
    try:
        from llcuda import _llcuda
        config = _llcuda.ModelConfig()
        assert config is not None
        
        # Test setting properties
        config.model_path = "/path/to/model.gguf"
        config.gpu_layers = 8
        config.ctx_size = 2048
        
        assert config.model_path == "/path/to/model.gguf"
        assert config.gpu_layers == 8
        assert config.ctx_size == 2048
    except ImportError:
        pytest.skip("C++ extension not built")


def test_engine_properties():
    """Test InferenceEngine properties"""
    import llcuda
    engine = llcuda.InferenceEngine()
    
    # Test is_loaded property
    assert hasattr(engine, 'is_loaded')
    assert isinstance(engine.is_loaded, bool)
    assert engine.is_loaded == False  # No model loaded initially


def test_quick_infer_function():
    """Test quick_infer convenience function"""
    import llcuda
    assert hasattr(llcuda, 'quick_infer')
    assert callable(llcuda.quick_infer)


@pytest.mark.skipif(
    not os.path.exists('/tmp/test_model.gguf'),
    reason="Test model not available"
)
def test_model_loading():
    """Test loading a model (requires actual model file)"""
    import llcuda
    engine = llcuda.InferenceEngine()
    
    # Create dummy model file for testing
    os.makedirs('/tmp', exist_ok=True)
    open('/tmp/test_model.gguf', 'a').close()
    
    result = engine.load_model('/tmp/test_model.gguf', gpu_layers=0)
    assert isinstance(result, bool)


def test_metrics_structure():
    """Test metrics dictionary structure"""
    import llcuda
    engine = llcuda.InferenceEngine()
    
    metrics = engine.get_metrics()
    assert isinstance(metrics, dict)
    assert 'latency' in metrics
    assert 'throughput' in metrics
    assert 'gpu' in metrics
    
    # Check latency metrics
    assert 'mean_ms' in metrics['latency']
    assert 'p50_ms' in metrics['latency']
    assert 'p95_ms' in metrics['latency']
    assert 'p99_ms' in metrics['latency']
    
    # Check throughput metrics
    assert 'total_tokens' in metrics['throughput']
    assert 'total_requests' in metrics['throughput']
    assert 'tokens_per_sec' in metrics['throughput']


def test_reset_metrics():
    """Test resetting metrics"""
    import llcuda
    engine = llcuda.InferenceEngine()
    
    # Should not raise any exception
    engine.reset_metrics()
    
    metrics = engine.get_metrics()
    assert metrics['throughput']['total_requests'] == 0
    assert metrics['throughput']['total_tokens'] == 0


def test_infer_result_wrapper():
    """Test InferResult wrapper class"""
    import llcuda
    
    # Create a mock result
    try:
        from llcuda import _llcuda
        cpp_result = _llcuda.InferResult()
        result = llcuda.InferResult(cpp_result)
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'text')
        assert hasattr(result, 'tokens_generated')
        assert hasattr(result, 'latency_ms')
        assert hasattr(result, 'tokens_per_sec')
        assert hasattr(result, 'error_message')
    except ImportError:
        pytest.skip("C++ extension not built")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
