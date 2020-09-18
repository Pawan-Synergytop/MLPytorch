import UIKit

class DataPredictor: Predictor {
    private var isRunning: Bool = false
    
    
    func predictData(_ buffer: [Float32], shape: [NSNumber],filePath:String) throws -> ([NSNumber], Double)? {
        
        if isRunning {
            return nil
        }
        let module: DataScienceTorchModule = {
               if let module = DataScienceTorchModule(fileAtPath: filePath) {
                return module
            } else {
                fatalError("Failed to load model!")
            }
        }()
        
        isRunning = true
        let startTime = CACurrentMediaTime()
        var tensorBuffer = buffer;
       
        guard let outputs = module.predictHlstmModel(data: UnsafeMutableRawPointer(&tensorBuffer),shap: shape) else {
            throw PredictorError.invalidInputTensor
        }
        isRunning = false
        let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
        return (outputs, inferenceTime)
    }
}
