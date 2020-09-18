#import "TorchModule.h"
#import <LibTorch/LibTorch.h>

@implementation TorchModule {
@protected
    torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _impl = torch::jit::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

@end






@implementation VisionTorchModule

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        auto outputTensor = _impl.forward({tensor}).toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 1000; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}


@end

@implementation DataScienceTorchModule

 - (NSArray<NSNumber *> *)predictData:(void *)csvDataBuffer shap:(NSArray<NSNumber *> *)shapBuffer{
    
//    at::IntArrayRef sizes;
//
//
//     int64_t myarray[] = {};
//
//       int i;
//       for(i=0;i<shapBuffer.count;i++) {
//           myarray[i] = (int64_t) shapBuffer[i].intValue;
//           printf("i=%d a[i]=%llu\n", i, myarray[i]);
//       }

     at::Tensor tensor;
        try {
            
            //at::Tensor tensor = torch::from_blob(csvDataBuffer, sizes, at::kFloat);
            
            if (shapBuffer.count == 1) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue)}, at::kFloat);
                
            }else if (shapBuffer.count == 2) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue)}, at::kFloat);
                
            }else if (shapBuffer.count == 3) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue), (int64_t)(shapBuffer[2].intValue)}, at::kFloat);

            }else if (shapBuffer.count == 4) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue), (int64_t)(shapBuffer[2].intValue), (int64_t)(shapBuffer[3].intValue)}, at::kFloat);

            }else if (shapBuffer.count == 5) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue), (int64_t)(shapBuffer[2].intValue), (int64_t)(shapBuffer[3].intValue), (int64_t)(shapBuffer[4].intValue)}, at::kFloat);

            }else if (shapBuffer.count == 6) {
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue), (int64_t)(shapBuffer[2].intValue), (int64_t)(shapBuffer[3].intValue), (int64_t)(shapBuffer[4].intValue), (int64_t)(shapBuffer[5].intValue)}, at::kFloat);

            }else{
                tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue), (int64_t)(shapBuffer[2].intValue), (int64_t)(shapBuffer[3].intValue), (int64_t)(shapBuffer[4].intValue), (int64_t)(shapBuffer[5].intValue), (int64_t)(shapBuffer[6].intValue)}, at::kFloat);
                
            }
          
            // tensor = torch::from_blob(csvDataBuffer, {(int64_t)(shapBuffer[0].intValue), (int64_t)(shapBuffer[1].intValue)}, at::kFloat);

            
            torch::autograd::AutoGradMode guard(false);
            at::AutoNonVariableTypeMode non_var_type_mode(true);
            auto outputTensor = _impl.forward({tensor}).toTensor();
            
            float* floatBuffer = outputTensor.data_ptr<float>();
            if (!floatBuffer) {
                return nil;
            }
            
            
            
            
            NSMutableArray* results = [[NSMutableArray alloc] init];
            for (int i = 0; i < 4; i++) {
                [results addObject:@(floatBuffer[i])];
//                NSLog(@"%f",floatBuffer[i]);
            }
             NSLog(@"%lu",(unsigned long)results.count);
            
            return [results copy];
            
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
        }
        return nil;
    }

    @end
    
    
    @implementation NLPTorchModule
    
    - (NSArray<NSNumber*>*)predictText:(NSString*)text {
        try {
            const char* buffer = text.UTF8String;
            torch::autograd::AutoGradMode guard(false);
            at::AutoNonVariableTypeMode non_var_type_mode(true);
            at::Tensor tensor = torch::from_blob((void*)buffer, {1, (int64_t)(strlen(buffer))}, at::kByte);
            auto outputTensor = _impl.forward({tensor}).toTensor();
            float* floatBuffer = outputTensor.data_ptr<float>();
            if (!floatBuffer) {
                return nil;
            }
            NSMutableArray* results = [[NSMutableArray alloc] init];
            for (int i = 0; i < 16; i++) {
                [results addObject:@(floatBuffer[i])];
            }
            return [results copy];
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
        }
        return nil;
    }
    
//    - (NSArray<NSString*>*)topics {
//        try {
//            auto genericList = _impl.run_method("get_classes").toGenericList();
//            NSMutableArray<NSString*>* topics = [NSMutableArray<NSString*> new];
//            for (int i = 0; i < genericList.size(); i++) {
//                std::string topic = genericList.get(i).toString()->string();
//                [topics addObject:[NSString stringWithCString:topic.c_str() encoding:NSUTF8StringEncoding]];
//            }
//            return [topics copy];
//        } catch (const std::exception& exception) {
//            NSLog(@"%s", exception.what());
//        }
//        return nil;
//    }
    
    @end
