/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// https://github.com/hailo-ai/hailort/blob/a987921b1fd4f09d76622cd9b49bc64fb814f299/hailort/libhailort/examples/cpp/async_infer_basic_example/async_infer_basic_example.cpp

#include <jni.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "hailo/hailort.hpp"

#if defined(__unix__)
#include <sys/mman.h>
#endif

// JNI class reference (this can be global since it's shared)
static jclass detectionResultClass = nullptr;
static jclass runtimeExceptionClass = nullptr;

void ThrowRuntimeException(JNIEnv *env, const char *message) {
  if (runtimeExceptionClass) {
    env->ThrowNew(runtimeExceptionClass, message);
  }
}

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
#if defined(__unix__)
  auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ,
                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (MAP_FAILED == addr)
    throw std::bad_alloc();
  return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(addr),
                                  [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
  auto addr =
      VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!addr)
    throw std::bad_alloc();
  return std::shared_ptr<uint8_t>(
      reinterpret_cast<uint8_t *>(addr),
      [](void *addr) { VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}

using namespace hailort;

struct HailoDetector {
  std::unique_ptr<VDevice> device;
  std::shared_ptr<InferModel> infer_model;
  ConfiguredInferModel configured_infer_model;
};

struct BOX_RECT {
  int left;
  int right;
  int top;
  int bottom;
};

struct DetectionResult {
  int id;
  BOX_RECT box;
  float obj_conf;
};

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Cache exception classes
  jclass localRuntimeClass = env->FindClass("java/lang/RuntimeException");
  if (localRuntimeClass) {
    runtimeExceptionClass = (jclass)env->NewGlobalRef(localRuntimeClass);
    env->DeleteLocalRef(localRuntimeClass);
  }

  // Find the detection result class
  jclass localClass =
      env->FindClass("org/photonvision/hailo/HailoJNI$HailoResult");
  if (!localClass) {
    std::printf(
        "Couldn't find class org/photonvision/hailo/HailoJNI$HailoResult!\n");
    return JNI_ERR;
  }

  // Create global reference
  detectionResultClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);

  if (!detectionResultClass) {
    std::printf("Couldn't create global reference to class!\n");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
    if (detectionResultClass) {
      env->DeleteGlobalRef(detectionResultClass);
      detectionResultClass = nullptr;
    }
  }
}

static jobject MakeJObject(JNIEnv *env, const DetectionResult &result) {
  if (!detectionResultClass) {
    std::printf("ERROR: detectionResultClass is null!\n");
    return nullptr;
  }

  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");
  if (!constructor) {
    std::printf("ERROR: Could not find constructor for HailoResult!\n");
    return nullptr;
  }

  return env->NewObject(detectionResultClass, constructor, result.box.left,
                        result.box.top, result.box.right, result.box.bottom,
                        result.obj_conf, result.id);
}

/*
 * Class:     org_photonvision_hailo_HailoJNI
 * Method:    create
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_hailo_HailoJNI_create
  (JNIEnv *env, jobject obj, jstring modelPath)
{
  try {
    jboolean isCopy;
    const char *convertedValue = (env)->GetStringUTFChars(modelPath, &isCopy);
    std::string model_path = convertedValue;
    (env)->ReleaseStringUTFChars(modelPath, convertedValue);
    hailort::Hef model =
        hailort::Hef::create(model_path).expect("Failed to load model");

    HailoDetector *detector = new HailoDetector;
    detector->device = VDevice::create().expect("Failed create vdevice");
    detector->infer_model = detector->device->create_infer_model(model).expect(
        "Failed to create infer model");
    detector->configured_infer_model = detector->infer_model->configure().expect(
        "Failed to create configured infer model");

    jlong ptr = reinterpret_cast<jlong>(detector);

    return ptr;
  } catch (const hailort_error &exception) {
    std::cout << "Failed to run inference. status=" << exception.status()
              << ", error message: " << exception.what() << std::endl;
    return -1;
  }
}
/*
 * Class:     org_photonvision_hailo_HailoJNI
 * Method:    delete
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT void JNICALL
Java_org_photonvision_hailo_HailoJNI_delete(JNIEnv *env, jlong detectorHandle) {
  HailoDetector *detector = reinterpret_cast<HailoDetector *>(detectorHandle);
  delete detector;
}

/*
 * Class:     org_photonvision_hailo_HailoJNI
 * Method:    detect
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_hailo_HailoJNI_detect(
    JNIEnv *env, jlong detectorHandle, jlong imageHandle, jdouble boxThreshold,
    jdouble nmsThreshold) {
  HailoDetector *detector = reinterpret_cast<HailoDetector *>(detectorHandle);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid HailoDetector pointer");
    return 0;
  }

  cv::Mat *inputImage = reinterpret_cast<cv::Mat *>(imageHandle);

  if (!inputImage) {
    ThrowRuntimeException(env, "Invalid input image pointer");
    return 0;
  }

  try {
    auto bindings = detector->configured_infer_model.create_bindings().expect(
        "Failed to create infer bindings");
    std::vector<std::shared_ptr<uint8_t>> buffer_guards;
    for (const auto &input_name : detector->infer_model->get_input_names()) {
      size_t input_frame_size =
          detector->infer_model->input(input_name)->get_frame_size();
      auto input_buffer = page_aligned_alloc(input_frame_size);
      auto status =
          bindings.input(input_name)
              ->set_buffer(MemoryView(input_buffer.get(), input_frame_size));
      if (HAILO_SUCCESS != status) {
        throw hailort_error(status, "Failed to set infer input buffer");
      }

      buffer_guards.push_back(input_buffer);
    }

    for (const auto &output_name : detector->infer_model->get_output_names()) {
      size_t output_frame_size =
          detector->infer_model->output(output_name)->get_frame_size();
      auto output_buffer = page_aligned_alloc(output_frame_size);
      auto status =
          bindings.output(output_name)
              ->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
      if (HAILO_SUCCESS != status) {
        throw hailort_error(status, "Failed to set infer output buffer");
      }

      buffer_guards.push_back(output_buffer);
    }

    // Run the async infer job
    auto job = detector->configured_infer_model.run_async(bindings).expect(
        "Failed to start async infer job");
    auto status = job.wait(std::chrono::milliseconds(10));
    if (HAILO_SUCCESS != status) {
      throw hailort_error(status, "Failed to wait for infer to finish");
    }
  } catch (const hailort_error &exception) {
    std::cout << "Failed to run inference. status=" << exception.status()
              << ", error message: " << exception.what() << std::endl;
    return -1;
  }

  return 0;
}
} // extern "C"
