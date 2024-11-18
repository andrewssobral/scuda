#define RPC___cudaRegisterVar 0
#define RPC___cudaRegisterFunction 1
#define RPC___cudaRegisterFatBinary 2
#define RPC___cudaRegisterFatBinaryEnd 3
#define RPC___cudaPushCallConfiguration 4
#define RPC___cudaPopCallConfiguration 5
#define RPC_nvmlInit_v2 6
#define RPC_nvmlInitWithFlags 7
#define RPC_nvmlShutdown 8
#define RPC_nvmlErrorString 9
#define RPC_nvmlSystemGetDriverVersion 10
#define RPC_nvmlSystemGetNVMLVersion 11
#define RPC_nvmlSystemGetCudaDriverVersion 12
#define RPC_nvmlSystemGetCudaDriverVersion_v2 13
#define RPC_nvmlSystemGetProcessName 14
#define RPC_nvmlUnitGetCount 15
#define RPC_nvmlUnitGetHandleByIndex 16
#define RPC_nvmlUnitGetUnitInfo 17
#define RPC_nvmlUnitGetLedState 18
#define RPC_nvmlUnitGetPsuInfo 19
#define RPC_nvmlUnitGetTemperature 20
#define RPC_nvmlUnitGetFanSpeedInfo 21
#define RPC_nvmlUnitGetDevices 22
#define RPC_nvmlSystemGetHicVersion 23
#define RPC_nvmlDeviceGetCount_v2 24
#define RPC_nvmlDeviceGetAttributes_v2 25
#define RPC_nvmlDeviceGetHandleByIndex_v2 26
#define RPC_nvmlDeviceGetHandleBySerial 27
#define RPC_nvmlDeviceGetHandleByUUID 28
#define RPC_nvmlDeviceGetHandleByPciBusId_v2 29
#define RPC_nvmlDeviceGetName 30
#define RPC_nvmlDeviceGetBrand 31
#define RPC_nvmlDeviceGetIndex 32
#define RPC_nvmlDeviceGetSerial 33
#define RPC_nvmlDeviceGetMemoryAffinity 34
#define RPC_nvmlDeviceGetCpuAffinityWithinScope 35
#define RPC_nvmlDeviceGetCpuAffinity 36
#define RPC_nvmlDeviceSetCpuAffinity 37
#define RPC_nvmlDeviceClearCpuAffinity 38
#define RPC_nvmlDeviceGetTopologyCommonAncestor 39
#define RPC_nvmlDeviceGetTopologyNearestGpus 40
#define RPC_nvmlSystemGetTopologyGpuSet 41
#define RPC_nvmlDeviceGetP2PStatus 42
#define RPC_nvmlDeviceGetUUID 43
#define RPC_nvmlVgpuInstanceGetMdevUUID 44
#define RPC_nvmlDeviceGetMinorNumber 45
#define RPC_nvmlDeviceGetBoardPartNumber 46
#define RPC_nvmlDeviceGetInforomVersion 47
#define RPC_nvmlDeviceGetInforomImageVersion 48
#define RPC_nvmlDeviceGetInforomConfigurationChecksum 49
#define RPC_nvmlDeviceValidateInforom 50
#define RPC_nvmlDeviceGetDisplayMode 51
#define RPC_nvmlDeviceGetDisplayActive 52
#define RPC_nvmlDeviceGetPersistenceMode 53
#define RPC_nvmlDeviceGetPciInfo_v3 54
#define RPC_nvmlDeviceGetMaxPcieLinkGeneration 55
#define RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration 56
#define RPC_nvmlDeviceGetMaxPcieLinkWidth 57
#define RPC_nvmlDeviceGetCurrPcieLinkGeneration 58
#define RPC_nvmlDeviceGetCurrPcieLinkWidth 59
#define RPC_nvmlDeviceGetPcieThroughput 60
#define RPC_nvmlDeviceGetPcieReplayCounter 61
#define RPC_nvmlDeviceGetClockInfo 62
#define RPC_nvmlDeviceGetMaxClockInfo 63
#define RPC_nvmlDeviceGetApplicationsClock 64
#define RPC_nvmlDeviceGetDefaultApplicationsClock 65
#define RPC_nvmlDeviceResetApplicationsClocks 66
#define RPC_nvmlDeviceGetClock 67
#define RPC_nvmlDeviceGetMaxCustomerBoostClock 68
#define RPC_nvmlDeviceGetSupportedMemoryClocks 69
#define RPC_nvmlDeviceGetSupportedGraphicsClocks 70
#define RPC_nvmlDeviceGetAutoBoostedClocksEnabled 71
#define RPC_nvmlDeviceSetAutoBoostedClocksEnabled 72
#define RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled 73
#define RPC_nvmlDeviceGetFanSpeed 74
#define RPC_nvmlDeviceGetFanSpeed_v2 75
#define RPC_nvmlDeviceGetTargetFanSpeed 76
#define RPC_nvmlDeviceSetDefaultFanSpeed_v2 77
#define RPC_nvmlDeviceGetMinMaxFanSpeed 78
#define RPC_nvmlDeviceGetFanControlPolicy_v2 79
#define RPC_nvmlDeviceSetFanControlPolicy 80
#define RPC_nvmlDeviceGetNumFans 81
#define RPC_nvmlDeviceGetTemperature 82
#define RPC_nvmlDeviceGetTemperatureThreshold 83
#define RPC_nvmlDeviceSetTemperatureThreshold 84
#define RPC_nvmlDeviceGetThermalSettings 85
#define RPC_nvmlDeviceGetPerformanceState 86
#define RPC_nvmlDeviceGetCurrentClocksThrottleReasons 87
#define RPC_nvmlDeviceGetSupportedClocksThrottleReasons 88
#define RPC_nvmlDeviceGetPowerState 89
#define RPC_nvmlDeviceGetPowerManagementMode 90
#define RPC_nvmlDeviceGetPowerManagementLimit 91
#define RPC_nvmlDeviceGetPowerManagementLimitConstraints 92
#define RPC_nvmlDeviceGetPowerManagementDefaultLimit 93
#define RPC_nvmlDeviceGetPowerUsage 94
#define RPC_nvmlDeviceGetTotalEnergyConsumption 95
#define RPC_nvmlDeviceGetEnforcedPowerLimit 96
#define RPC_nvmlDeviceGetGpuOperationMode 97
#define RPC_nvmlDeviceGetMemoryInfo 98
#define RPC_nvmlDeviceGetMemoryInfo_v2 99
#define RPC_nvmlDeviceGetComputeMode 100
#define RPC_nvmlDeviceGetCudaComputeCapability 101
#define RPC_nvmlDeviceGetEccMode 102
#define RPC_nvmlDeviceGetDefaultEccMode 103
#define RPC_nvmlDeviceGetBoardId 104
#define RPC_nvmlDeviceGetMultiGpuBoard 105
#define RPC_nvmlDeviceGetTotalEccErrors 106
#define RPC_nvmlDeviceGetDetailedEccErrors 107
#define RPC_nvmlDeviceGetMemoryErrorCounter 108
#define RPC_nvmlDeviceGetUtilizationRates 109
#define RPC_nvmlDeviceGetEncoderUtilization 110
#define RPC_nvmlDeviceGetEncoderCapacity 111
#define RPC_nvmlDeviceGetEncoderStats 112
#define RPC_nvmlDeviceGetEncoderSessions 113
#define RPC_nvmlDeviceGetDecoderUtilization 114
#define RPC_nvmlDeviceGetFBCStats 115
#define RPC_nvmlDeviceGetFBCSessions 116
#define RPC_nvmlDeviceGetDriverModel 117
#define RPC_nvmlDeviceGetVbiosVersion 118
#define RPC_nvmlDeviceGetBridgeChipInfo 119
#define RPC_nvmlDeviceGetComputeRunningProcesses_v3 120
#define RPC_nvmlDeviceGetGraphicsRunningProcesses_v3 121
#define RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3 122
#define RPC_nvmlDeviceOnSameBoard 123
#define RPC_nvmlDeviceGetAPIRestriction 124
#define RPC_nvmlDeviceGetSamples 125
#define RPC_nvmlDeviceGetBAR1MemoryInfo 126
#define RPC_nvmlDeviceGetViolationStatus 127
#define RPC_nvmlDeviceGetIrqNum 128
#define RPC_nvmlDeviceGetNumGpuCores 129
#define RPC_nvmlDeviceGetPowerSource 130
#define RPC_nvmlDeviceGetMemoryBusWidth 131
#define RPC_nvmlDeviceGetPcieLinkMaxSpeed 132
#define RPC_nvmlDeviceGetPcieSpeed 133
#define RPC_nvmlDeviceGetAdaptiveClockInfoStatus 134
#define RPC_nvmlDeviceGetAccountingMode 135
#define RPC_nvmlDeviceGetAccountingStats 136
#define RPC_nvmlDeviceGetAccountingPids 137
#define RPC_nvmlDeviceGetAccountingBufferSize 138
#define RPC_nvmlDeviceGetRetiredPages 139
#define RPC_nvmlDeviceGetRetiredPages_v2 140
#define RPC_nvmlDeviceGetRetiredPagesPendingStatus 141
#define RPC_nvmlDeviceGetRemappedRows 142
#define RPC_nvmlDeviceGetRowRemapperHistogram 143
#define RPC_nvmlDeviceGetArchitecture 144
#define RPC_nvmlUnitSetLedState 145
#define RPC_nvmlDeviceSetPersistenceMode 146
#define RPC_nvmlDeviceSetComputeMode 147
#define RPC_nvmlDeviceSetEccMode 148
#define RPC_nvmlDeviceClearEccErrorCounts 149
#define RPC_nvmlDeviceSetDriverModel 150
#define RPC_nvmlDeviceSetGpuLockedClocks 151
#define RPC_nvmlDeviceResetGpuLockedClocks 152
#define RPC_nvmlDeviceSetMemoryLockedClocks 153
#define RPC_nvmlDeviceResetMemoryLockedClocks 154
#define RPC_nvmlDeviceSetApplicationsClocks 155
#define RPC_nvmlDeviceGetClkMonStatus 156
#define RPC_nvmlDeviceSetPowerManagementLimit 157
#define RPC_nvmlDeviceSetGpuOperationMode 158
#define RPC_nvmlDeviceSetAPIRestriction 159
#define RPC_nvmlDeviceSetAccountingMode 160
#define RPC_nvmlDeviceClearAccountingPids 161
#define RPC_nvmlDeviceGetNvLinkState 162
#define RPC_nvmlDeviceGetNvLinkVersion 163
#define RPC_nvmlDeviceGetNvLinkCapability 164
#define RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2 165
#define RPC_nvmlDeviceGetNvLinkErrorCounter 166
#define RPC_nvmlDeviceResetNvLinkErrorCounters 167
#define RPC_nvmlDeviceSetNvLinkUtilizationControl 168
#define RPC_nvmlDeviceGetNvLinkUtilizationControl 169
#define RPC_nvmlDeviceGetNvLinkUtilizationCounter 170
#define RPC_nvmlDeviceFreezeNvLinkUtilizationCounter 171
#define RPC_nvmlDeviceResetNvLinkUtilizationCounter 172
#define RPC_nvmlDeviceGetNvLinkRemoteDeviceType 173
#define RPC_nvmlEventSetCreate 174
#define RPC_nvmlDeviceRegisterEvents 175
#define RPC_nvmlDeviceGetSupportedEventTypes 176
#define RPC_nvmlEventSetWait_v2 177
#define RPC_nvmlEventSetFree 178
#define RPC_nvmlDeviceModifyDrainState 179
#define RPC_nvmlDeviceQueryDrainState 180
#define RPC_nvmlDeviceRemoveGpu_v2 181
#define RPC_nvmlDeviceDiscoverGpus 182
#define RPC_nvmlDeviceGetFieldValues 183
#define RPC_nvmlDeviceClearFieldValues 184
#define RPC_nvmlDeviceGetVirtualizationMode 185
#define RPC_nvmlDeviceGetHostVgpuMode 186
#define RPC_nvmlDeviceSetVirtualizationMode 187
#define RPC_nvmlDeviceGetGridLicensableFeatures_v4 188
#define RPC_nvmlDeviceGetProcessUtilization 189
#define RPC_nvmlDeviceGetGspFirmwareVersion 190
#define RPC_nvmlDeviceGetGspFirmwareMode 191
#define RPC_nvmlGetVgpuDriverCapabilities 192
#define RPC_nvmlDeviceGetVgpuCapabilities 193
#define RPC_nvmlDeviceGetSupportedVgpus 194
#define RPC_nvmlDeviceGetCreatableVgpus 195
#define RPC_nvmlVgpuTypeGetClass 196
#define RPC_nvmlVgpuTypeGetName 197
#define RPC_nvmlVgpuTypeGetGpuInstanceProfileId 198
#define RPC_nvmlVgpuTypeGetDeviceID 199
#define RPC_nvmlVgpuTypeGetFramebufferSize 200
#define RPC_nvmlVgpuTypeGetNumDisplayHeads 201
#define RPC_nvmlVgpuTypeGetResolution 202
#define RPC_nvmlVgpuTypeGetLicense 203
#define RPC_nvmlVgpuTypeGetFrameRateLimit 204
#define RPC_nvmlVgpuTypeGetMaxInstances 205
#define RPC_nvmlVgpuTypeGetMaxInstancesPerVm 206
#define RPC_nvmlDeviceGetActiveVgpus 207
#define RPC_nvmlVgpuInstanceGetVmID 208
#define RPC_nvmlVgpuInstanceGetUUID 209
#define RPC_nvmlVgpuInstanceGetVmDriverVersion 210
#define RPC_nvmlVgpuInstanceGetFbUsage 211
#define RPC_nvmlVgpuInstanceGetLicenseStatus 212
#define RPC_nvmlVgpuInstanceGetType 213
#define RPC_nvmlVgpuInstanceGetFrameRateLimit 214
#define RPC_nvmlVgpuInstanceGetEccMode 215
#define RPC_nvmlVgpuInstanceGetEncoderCapacity 216
#define RPC_nvmlVgpuInstanceSetEncoderCapacity 217
#define RPC_nvmlVgpuInstanceGetEncoderStats 218
#define RPC_nvmlVgpuInstanceGetEncoderSessions 219
#define RPC_nvmlVgpuInstanceGetFBCStats 220
#define RPC_nvmlVgpuInstanceGetFBCSessions 221
#define RPC_nvmlVgpuInstanceGetGpuInstanceId 222
#define RPC_nvmlVgpuInstanceGetGpuPciId 223
#define RPC_nvmlVgpuTypeGetCapabilities 224
#define RPC_nvmlVgpuInstanceGetMetadata 225
#define RPC_nvmlDeviceGetVgpuMetadata 226
#define RPC_nvmlGetVgpuCompatibility 227
#define RPC_nvmlDeviceGetPgpuMetadataString 228
#define RPC_nvmlDeviceGetVgpuSchedulerLog 229
#define RPC_nvmlDeviceGetVgpuSchedulerState 230
#define RPC_nvmlDeviceGetVgpuSchedulerCapabilities 231
#define RPC_nvmlGetVgpuVersion 232
#define RPC_nvmlSetVgpuVersion 233
#define RPC_nvmlDeviceGetVgpuUtilization 234
#define RPC_nvmlDeviceGetVgpuProcessUtilization 235
#define RPC_nvmlVgpuInstanceGetAccountingMode 236
#define RPC_nvmlVgpuInstanceGetAccountingPids 237
#define RPC_nvmlVgpuInstanceGetAccountingStats 238
#define RPC_nvmlVgpuInstanceClearAccountingPids 239
#define RPC_nvmlVgpuInstanceGetLicenseInfo_v2 240
#define RPC_nvmlGetExcludedDeviceCount 241
#define RPC_nvmlGetExcludedDeviceInfoByIndex 242
#define RPC_nvmlDeviceSetMigMode 243
#define RPC_nvmlDeviceGetMigMode 244
#define RPC_nvmlDeviceGetGpuInstanceProfileInfo 245
#define RPC_nvmlDeviceGetGpuInstanceProfileInfoV 246
#define RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2 247
#define RPC_nvmlDeviceGetGpuInstanceRemainingCapacity 248
#define RPC_nvmlDeviceCreateGpuInstance 249
#define RPC_nvmlDeviceCreateGpuInstanceWithPlacement 250
#define RPC_nvmlGpuInstanceDestroy 251
#define RPC_nvmlDeviceGetGpuInstances 252
#define RPC_nvmlDeviceGetGpuInstanceById 253
#define RPC_nvmlGpuInstanceGetInfo 254
#define RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo 255
#define RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV 256
#define RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity 257
#define RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements 258
#define RPC_nvmlGpuInstanceCreateComputeInstance 259
#define RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement 260
#define RPC_nvmlComputeInstanceDestroy 261
#define RPC_nvmlGpuInstanceGetComputeInstances 262
#define RPC_nvmlGpuInstanceGetComputeInstanceById 263
#define RPC_nvmlComputeInstanceGetInfo_v2 264
#define RPC_nvmlDeviceIsMigDeviceHandle 265
#define RPC_nvmlDeviceGetGpuInstanceId 266
#define RPC_nvmlDeviceGetComputeInstanceId 267
#define RPC_nvmlDeviceGetMaxMigDeviceCount 268
#define RPC_nvmlDeviceGetMigDeviceHandleByIndex 269
#define RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle 270
#define RPC_nvmlDeviceGetBusType 271
#define RPC_nvmlDeviceGetDynamicPstatesInfo 272
#define RPC_nvmlDeviceSetFanSpeed_v2 273
#define RPC_nvmlDeviceGetGpcClkVfOffset 274
#define RPC_nvmlDeviceSetGpcClkVfOffset 275
#define RPC_nvmlDeviceGetMemClkVfOffset 276
#define RPC_nvmlDeviceSetMemClkVfOffset 277
#define RPC_nvmlDeviceGetMinMaxClockOfPState 278
#define RPC_nvmlDeviceGetSupportedPerformanceStates 279
#define RPC_nvmlDeviceGetGpcClkMinMaxVfOffset 280
#define RPC_nvmlDeviceGetMemClkMinMaxVfOffset 281
#define RPC_nvmlDeviceGetGpuFabricInfo 282
#define RPC_nvmlGpmMetricsGet 283
#define RPC_nvmlGpmSampleFree 284
#define RPC_nvmlGpmSampleAlloc 285
#define RPC_nvmlGpmSampleGet 286
#define RPC_nvmlGpmMigSampleGet 287
#define RPC_nvmlGpmQueryDeviceSupport 288
#define RPC_nvmlDeviceCcuGetStreamState 289
#define RPC_nvmlDeviceCcuSetStreamState 290
#define RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold 291
#define RPC_cuGetErrorString 292
#define RPC_cuGetErrorName 293
#define RPC_cuInit 294
#define RPC_cuDriverGetVersion 295
#define RPC_cuDeviceGet 296
#define RPC_cuDeviceGetCount 297
#define RPC_cuDeviceGetName 298
#define RPC_cuDeviceGetUuid 299
#define RPC_cuDeviceGetUuid_v2 300
#define RPC_cuDeviceGetLuid 301
#define RPC_cuDeviceTotalMem_v2 302
#define RPC_cuDeviceGetTexture1DLinearMaxWidth 303
#define RPC_cuDeviceGetAttribute 304
#define RPC_cuDeviceGetNvSciSyncAttributes 305
#define RPC_cuDeviceSetMemPool 306
#define RPC_cuDeviceGetMemPool 307
#define RPC_cuDeviceGetDefaultMemPool 308
#define RPC_cuDeviceGetExecAffinitySupport 309
#define RPC_cuFlushGPUDirectRDMAWrites 310
#define RPC_cuDeviceGetProperties 311
#define RPC_cuDeviceComputeCapability 312
#define RPC_cuDevicePrimaryCtxRetain 313
#define RPC_cuDevicePrimaryCtxRelease_v2 314
#define RPC_cuDevicePrimaryCtxSetFlags_v2 315
#define RPC_cuDevicePrimaryCtxGetState 316
#define RPC_cuDevicePrimaryCtxReset_v2 317
#define RPC_cuCtxCreate_v2 318
#define RPC_cuCtxCreate_v3 319
#define RPC_cuCtxDestroy_v2 320
#define RPC_cuCtxPushCurrent_v2 321
#define RPC_cuCtxPopCurrent_v2 322
#define RPC_cuCtxSetCurrent 323
#define RPC_cuCtxGetCurrent 324
#define RPC_cuCtxGetDevice 325
#define RPC_cuCtxGetFlags 326
#define RPC_cuCtxGetId 327
#define RPC_cuCtxSynchronize 328
#define RPC_cuCtxSetLimit 329
#define RPC_cuCtxGetLimit 330
#define RPC_cuCtxGetCacheConfig 331
#define RPC_cuCtxSetCacheConfig 332
#define RPC_cuCtxGetSharedMemConfig 333
#define RPC_cuCtxSetSharedMemConfig 334
#define RPC_cuCtxGetApiVersion 335
#define RPC_cuCtxGetStreamPriorityRange 336
#define RPC_cuCtxResetPersistingL2Cache 337
#define RPC_cuCtxGetExecAffinity 338
#define RPC_cuCtxAttach 339
#define RPC_cuCtxDetach 340
#define RPC_cuModuleLoad 341
#define RPC_cuModuleLoadData 342
#define RPC_cuModuleLoadDataEx 343
#define RPC_cuModuleLoadFatBinary 344
#define RPC_cuModuleUnload 345
#define RPC_cuModuleGetLoadingMode 346
#define RPC_cuModuleGetFunction 347
#define RPC_cuModuleGetGlobal_v2 348
#define RPC_cuLinkCreate_v2 349
#define RPC_cuLinkAddFile_v2 350
#define RPC_cuLinkComplete 351
#define RPC_cuLinkDestroy 352
#define RPC_cuModuleGetTexRef 353
#define RPC_cuModuleGetSurfRef 354
#define RPC_cuLibraryLoadData 355
#define RPC_cuLibraryLoadFromFile 356
#define RPC_cuLibraryUnload 357
#define RPC_cuLibraryGetKernel 358
#define RPC_cuLibraryGetModule 359
#define RPC_cuKernelGetFunction 360
#define RPC_cuLibraryGetGlobal 361
#define RPC_cuLibraryGetManaged 362
#define RPC_cuLibraryGetUnifiedFunction 363
#define RPC_cuKernelGetAttribute 364
#define RPC_cuKernelSetAttribute 365
#define RPC_cuKernelSetCacheConfig 366
#define RPC_cuMemGetInfo_v2 367
#define RPC_cuMemAlloc_v2 368
#define RPC_cuMemAllocPitch_v2 369
#define RPC_cuMemFree_v2 370
#define RPC_cuMemGetAddressRange_v2 371
#define RPC_cuMemAllocHost_v2 372
#define RPC_cuMemFreeHost 373
#define RPC_cuMemHostAlloc 374
#define RPC_cuMemHostGetDevicePointer_v2 375
#define RPC_cuMemHostGetFlags 376
#define RPC_cuMemAllocManaged 377
#define RPC_cuDeviceGetByPCIBusId 378
#define RPC_cuDeviceGetPCIBusId 379
#define RPC_cuIpcGetEventHandle 380
#define RPC_cuIpcOpenEventHandle 381
#define RPC_cuIpcGetMemHandle 382
#define RPC_cuIpcOpenMemHandle_v2 383
#define RPC_cuIpcCloseMemHandle 384
#define RPC_cuMemcpy 385
#define RPC_cuMemcpyPeer 386
#define RPC_cuMemcpyHtoD_v2 387
#define RPC_cuMemcpyDtoD_v2 388
#define RPC_cuMemcpyDtoA_v2 389
#define RPC_cuMemcpyAtoD_v2 390
#define RPC_cuMemcpyHtoA_v2 391
#define RPC_cuMemcpyAtoH_v2 392
#define RPC_cuMemcpyAtoA_v2 393
#define RPC_cuMemcpy2D_v2 394
#define RPC_cuMemcpy2DUnaligned_v2 395
#define RPC_cuMemcpy3D_v2 396
#define RPC_cuMemcpy3DPeer 397
#define RPC_cuMemcpyAsync 398
#define RPC_cuMemcpyPeerAsync 399
#define RPC_cuMemcpyHtoDAsync_v2 400
#define RPC_cuMemcpyDtoDAsync_v2 401
#define RPC_cuMemcpyHtoAAsync_v2 402
#define RPC_cuMemcpy2DAsync_v2 403
#define RPC_cuMemcpy3DAsync_v2 404
#define RPC_cuMemcpy3DPeerAsync 405
#define RPC_cuMemsetD8_v2 406
#define RPC_cuMemsetD16_v2 407
#define RPC_cuMemsetD32_v2 408
#define RPC_cuMemsetD2D8_v2 409
#define RPC_cuMemsetD2D16_v2 410
#define RPC_cuMemsetD2D32_v2 411
#define RPC_cuMemsetD8Async 412
#define RPC_cuMemsetD16Async 413
#define RPC_cuMemsetD32Async 414
#define RPC_cuMemsetD2D8Async 415
#define RPC_cuMemsetD2D16Async 416
#define RPC_cuMemsetD2D32Async 417
#define RPC_cuArrayCreate_v2 418
#define RPC_cuArrayGetDescriptor_v2 419
#define RPC_cuArrayGetSparseProperties 420
#define RPC_cuMipmappedArrayGetSparseProperties 421
#define RPC_cuArrayGetMemoryRequirements 422
#define RPC_cuMipmappedArrayGetMemoryRequirements 423
#define RPC_cuArrayGetPlane 424
#define RPC_cuArrayDestroy 425
#define RPC_cuArray3DCreate_v2 426
#define RPC_cuArray3DGetDescriptor_v2 427
#define RPC_cuMipmappedArrayCreate 428
#define RPC_cuMipmappedArrayGetLevel 429
#define RPC_cuMipmappedArrayDestroy 430
#define RPC_cuMemAddressReserve 431
#define RPC_cuMemAddressFree 432
#define RPC_cuMemCreate 433
#define RPC_cuMemRelease 434
#define RPC_cuMemMap 435
#define RPC_cuMemMapArrayAsync 436
#define RPC_cuMemUnmap 437
#define RPC_cuMemSetAccess 438
#define RPC_cuMemGetAccess 439
#define RPC_cuMemGetAllocationGranularity 440
#define RPC_cuMemGetAllocationPropertiesFromHandle 441
#define RPC_cuMemFreeAsync 442
#define RPC_cuMemAllocAsync 443
#define RPC_cuMemPoolTrimTo 444
#define RPC_cuMemPoolSetAccess 445
#define RPC_cuMemPoolGetAccess 446
#define RPC_cuMemPoolCreate 447
#define RPC_cuMemPoolDestroy 448
#define RPC_cuMemAllocFromPoolAsync 449
#define RPC_cuMemPoolExportPointer 450
#define RPC_cuMemPoolImportPointer 451
#define RPC_cuMemPrefetchAsync 452
#define RPC_cuMemAdvise 453
#define RPC_cuMemRangeGetAttributes 454
#define RPC_cuPointerSetAttribute 455
#define RPC_cuPointerGetAttributes 456
#define RPC_cuStreamCreate 457
#define RPC_cuStreamCreateWithPriority 458
#define RPC_cuStreamGetPriority 459
#define RPC_cuStreamGetFlags 460
#define RPC_cuStreamGetId 461
#define RPC_cuStreamGetCtx 462
#define RPC_cuStreamWaitEvent 463
#define RPC_cuStreamBeginCapture_v2 464
#define RPC_cuThreadExchangeStreamCaptureMode 465
#define RPC_cuStreamEndCapture 466
#define RPC_cuStreamIsCapturing 467
#define RPC_cuStreamGetCaptureInfo_v2 468
#define RPC_cuStreamUpdateCaptureDependencies 469
#define RPC_cuStreamAttachMemAsync 470
#define RPC_cuStreamQuery 471
#define RPC_cuStreamSynchronize 472
#define RPC_cuStreamDestroy_v2 473
#define RPC_cuStreamCopyAttributes 474
#define RPC_cuStreamGetAttribute 475
#define RPC_cuStreamSetAttribute 476
#define RPC_cuEventCreate 477
#define RPC_cuEventRecord 478
#define RPC_cuEventRecordWithFlags 479
#define RPC_cuEventQuery 480
#define RPC_cuEventSynchronize 481
#define RPC_cuEventDestroy_v2 482
#define RPC_cuEventElapsedTime 483
#define RPC_cuImportExternalMemory 484
#define RPC_cuExternalMemoryGetMappedBuffer 485
#define RPC_cuExternalMemoryGetMappedMipmappedArray 486
#define RPC_cuDestroyExternalMemory 487
#define RPC_cuImportExternalSemaphore 488
#define RPC_cuSignalExternalSemaphoresAsync 489
#define RPC_cuWaitExternalSemaphoresAsync 490
#define RPC_cuDestroyExternalSemaphore 491
#define RPC_cuStreamWaitValue32_v2 492
#define RPC_cuStreamWaitValue64_v2 493
#define RPC_cuStreamWriteValue32_v2 494
#define RPC_cuStreamWriteValue64_v2 495
#define RPC_cuStreamBatchMemOp_v2 496
#define RPC_cuFuncGetAttribute 497
#define RPC_cuFuncSetAttribute 498
#define RPC_cuFuncSetCacheConfig 499
#define RPC_cuFuncSetSharedMemConfig 500
#define RPC_cuFuncGetModule 501
#define RPC_cuLaunchKernel 502
#define RPC_cuLaunchKernelEx 503
#define RPC_cuLaunchCooperativeKernel 504
#define RPC_cuLaunchCooperativeKernelMultiDevice 505
#define RPC_cuFuncSetBlockShape 506
#define RPC_cuFuncSetSharedSize 507
#define RPC_cuParamSetSize 508
#define RPC_cuParamSeti 509
#define RPC_cuParamSetf 510
#define RPC_cuLaunch 511
#define RPC_cuLaunchGrid 512
#define RPC_cuLaunchGridAsync 513
#define RPC_cuParamSetTexRef 514
#define RPC_cuGraphCreate 515
#define RPC_cuGraphAddKernelNode_v2 516
#define RPC_cuGraphKernelNodeGetParams_v2 517
#define RPC_cuGraphKernelNodeSetParams_v2 518
#define RPC_cuGraphAddMemcpyNode 519
#define RPC_cuGraphMemcpyNodeGetParams 520
#define RPC_cuGraphMemcpyNodeSetParams 521
#define RPC_cuGraphAddMemsetNode 522
#define RPC_cuGraphMemsetNodeGetParams 523
#define RPC_cuGraphMemsetNodeSetParams 524
#define RPC_cuGraphAddHostNode 525
#define RPC_cuGraphHostNodeGetParams 526
#define RPC_cuGraphHostNodeSetParams 527
#define RPC_cuGraphAddChildGraphNode 528
#define RPC_cuGraphChildGraphNodeGetGraph 529
#define RPC_cuGraphAddEmptyNode 530
#define RPC_cuGraphAddEventRecordNode 531
#define RPC_cuGraphEventRecordNodeGetEvent 532
#define RPC_cuGraphEventRecordNodeSetEvent 533
#define RPC_cuGraphAddEventWaitNode 534
#define RPC_cuGraphEventWaitNodeGetEvent 535
#define RPC_cuGraphEventWaitNodeSetEvent 536
#define RPC_cuGraphAddExternalSemaphoresSignalNode 537
#define RPC_cuGraphExternalSemaphoresSignalNodeGetParams 538
#define RPC_cuGraphExternalSemaphoresSignalNodeSetParams 539
#define RPC_cuGraphAddExternalSemaphoresWaitNode 540
#define RPC_cuGraphExternalSemaphoresWaitNodeGetParams 541
#define RPC_cuGraphExternalSemaphoresWaitNodeSetParams 542
#define RPC_cuGraphAddBatchMemOpNode 543
#define RPC_cuGraphBatchMemOpNodeGetParams 544
#define RPC_cuGraphBatchMemOpNodeSetParams 545
#define RPC_cuGraphExecBatchMemOpNodeSetParams 546
#define RPC_cuGraphAddMemAllocNode 547
#define RPC_cuGraphMemAllocNodeGetParams 548
#define RPC_cuGraphAddMemFreeNode 549
#define RPC_cuGraphMemFreeNodeGetParams 550
#define RPC_cuDeviceGraphMemTrim 551
#define RPC_cuGraphClone 552
#define RPC_cuGraphNodeFindInClone 553
#define RPC_cuGraphNodeGetType 554
#define RPC_cuGraphGetNodes 555
#define RPC_cuGraphGetRootNodes 556
#define RPC_cuGraphGetEdges 557
#define RPC_cuGraphNodeGetDependencies 558
#define RPC_cuGraphNodeGetDependentNodes 559
#define RPC_cuGraphAddDependencies 560
#define RPC_cuGraphRemoveDependencies 561
#define RPC_cuGraphDestroyNode 562
#define RPC_cuGraphInstantiateWithFlags 563
#define RPC_cuGraphInstantiateWithParams 564
#define RPC_cuGraphExecGetFlags 565
#define RPC_cuGraphExecKernelNodeSetParams_v2 566
#define RPC_cuGraphExecMemcpyNodeSetParams 567
#define RPC_cuGraphExecMemsetNodeSetParams 568
#define RPC_cuGraphExecHostNodeSetParams 569
#define RPC_cuGraphExecChildGraphNodeSetParams 570
#define RPC_cuGraphExecEventRecordNodeSetEvent 571
#define RPC_cuGraphExecEventWaitNodeSetEvent 572
#define RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams 573
#define RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams 574
#define RPC_cuGraphNodeSetEnabled 575
#define RPC_cuGraphNodeGetEnabled 576
#define RPC_cuGraphUpload 577
#define RPC_cuGraphLaunch 578
#define RPC_cuGraphExecDestroy 579
#define RPC_cuGraphDestroy 580
#define RPC_cuGraphExecUpdate_v2 581
#define RPC_cuGraphKernelNodeCopyAttributes 582
#define RPC_cuGraphKernelNodeGetAttribute 583
#define RPC_cuGraphKernelNodeSetAttribute 584
#define RPC_cuGraphDebugDotPrint 585
#define RPC_cuUserObjectRetain 586
#define RPC_cuUserObjectRelease 587
#define RPC_cuGraphRetainUserObject 588
#define RPC_cuGraphReleaseUserObject 589
#define RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor 590
#define RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags 591
#define RPC_cuOccupancyMaxPotentialBlockSize 592
#define RPC_cuOccupancyMaxPotentialBlockSizeWithFlags 593
#define RPC_cuOccupancyAvailableDynamicSMemPerBlock 594
#define RPC_cuOccupancyMaxPotentialClusterSize 595
#define RPC_cuOccupancyMaxActiveClusters 596
#define RPC_cuTexRefSetArray 597
#define RPC_cuTexRefSetMipmappedArray 598
#define RPC_cuTexRefSetAddress_v2 599
#define RPC_cuTexRefSetAddress2D_v3 600
#define RPC_cuTexRefSetFormat 601
#define RPC_cuTexRefSetAddressMode 602
#define RPC_cuTexRefSetFilterMode 603
#define RPC_cuTexRefSetMipmapFilterMode 604
#define RPC_cuTexRefSetMipmapLevelBias 605
#define RPC_cuTexRefSetMipmapLevelClamp 606
#define RPC_cuTexRefSetMaxAnisotropy 607
#define RPC_cuTexRefSetBorderColor 608
#define RPC_cuTexRefSetFlags 609
#define RPC_cuTexRefGetAddress_v2 610
#define RPC_cuTexRefGetArray 611
#define RPC_cuTexRefGetMipmappedArray 612
#define RPC_cuTexRefGetAddressMode 613
#define RPC_cuTexRefGetFilterMode 614
#define RPC_cuTexRefGetFormat 615
#define RPC_cuTexRefGetMipmapFilterMode 616
#define RPC_cuTexRefGetMipmapLevelBias 617
#define RPC_cuTexRefGetMipmapLevelClamp 618
#define RPC_cuTexRefGetMaxAnisotropy 619
#define RPC_cuTexRefGetBorderColor 620
#define RPC_cuTexRefGetFlags 621
#define RPC_cuTexRefCreate 622
#define RPC_cuTexRefDestroy 623
#define RPC_cuSurfRefSetArray 624
#define RPC_cuSurfRefGetArray 625
#define RPC_cuTexObjectCreate 626
#define RPC_cuTexObjectDestroy 627
#define RPC_cuTexObjectGetResourceDesc 628
#define RPC_cuTexObjectGetTextureDesc 629
#define RPC_cuTexObjectGetResourceViewDesc 630
#define RPC_cuSurfObjectCreate 631
#define RPC_cuSurfObjectDestroy 632
#define RPC_cuSurfObjectGetResourceDesc 633
#define RPC_cuDeviceCanAccessPeer 634
#define RPC_cuCtxEnablePeerAccess 635
#define RPC_cuCtxDisablePeerAccess 636
#define RPC_cuDeviceGetP2PAttribute 637
#define RPC_cuGraphicsUnregisterResource 638
#define RPC_cuGraphicsSubResourceGetMappedArray 639
#define RPC_cuGraphicsResourceGetMappedMipmappedArray 640
#define RPC_cuGraphicsResourceGetMappedPointer_v2 641
#define RPC_cuGraphicsResourceSetMapFlags_v2 642
#define RPC_cuGraphicsMapResources 643
#define RPC_cuGraphicsUnmapResources 644
#define RPC_cuGetProcAddress_v2 645
#define RPC_cuGetExportTable 646
#define RPC_cudaDeviceReset 647
#define RPC_cudaDeviceSynchronize 648
#define RPC_cudaDeviceSetLimit 649
#define RPC_cudaDeviceGetLimit 650
#define RPC_cudaDeviceGetTexture1DLinearMaxWidth 651
#define RPC_cudaDeviceGetCacheConfig 652
#define RPC_cudaDeviceGetStreamPriorityRange 653
#define RPC_cudaDeviceSetCacheConfig 654
#define RPC_cudaDeviceGetSharedMemConfig 655
#define RPC_cudaDeviceSetSharedMemConfig 656
#define RPC_cudaDeviceGetByPCIBusId 657
#define RPC_cudaDeviceGetPCIBusId 658
#define RPC_cudaIpcGetEventHandle 659
#define RPC_cudaIpcOpenEventHandle 660
#define RPC_cudaIpcOpenMemHandle 661
#define RPC_cudaDeviceFlushGPUDirectRDMAWrites 662
#define RPC_cudaThreadExit 663
#define RPC_cudaThreadSynchronize 664
#define RPC_cudaThreadSetLimit 665
#define RPC_cudaThreadGetLimit 666
#define RPC_cudaThreadGetCacheConfig 667
#define RPC_cudaThreadSetCacheConfig 668
#define RPC_cudaGetLastError 669
#define RPC_cudaPeekAtLastError 670
#define RPC_cudaGetErrorName 671
#define RPC_cudaGetErrorString 672
#define RPC_cudaGetDeviceCount 673
#define RPC_cudaGetDeviceProperties_v2 674
#define RPC_cudaDeviceGetAttribute 675
#define RPC_cudaDeviceGetDefaultMemPool 676
#define RPC_cudaDeviceSetMemPool 677
#define RPC_cudaDeviceGetMemPool 678
#define RPC_cudaDeviceGetP2PAttribute 679
#define RPC_cudaChooseDevice 680
#define RPC_cudaInitDevice 681
#define RPC_cudaSetDevice 682
#define RPC_cudaGetDevice 683
#define RPC_cudaSetValidDevices 684
#define RPC_cudaSetDeviceFlags 685
#define RPC_cudaGetDeviceFlags 686
#define RPC_cudaStreamCreate 687
#define RPC_cudaStreamCreateWithFlags 688
#define RPC_cudaStreamCreateWithPriority 689
#define RPC_cudaStreamGetPriority 690
#define RPC_cudaStreamGetFlags 691
#define RPC_cudaStreamGetId 692
#define RPC_cudaCtxResetPersistingL2Cache 693
#define RPC_cudaStreamCopyAttributes 694
#define RPC_cudaStreamGetAttribute 695
#define RPC_cudaStreamSetAttribute 696
#define RPC_cudaStreamDestroy 697
#define RPC_cudaStreamWaitEvent 698
#define RPC_cudaStreamSynchronize 699
#define RPC_cudaStreamQuery 700
#define RPC_cudaStreamBeginCapture 701
#define RPC_cudaThreadExchangeStreamCaptureMode 702
#define RPC_cudaStreamEndCapture 703
#define RPC_cudaStreamIsCapturing 704
#define RPC_cudaStreamGetCaptureInfo_v2 705
#define RPC_cudaStreamUpdateCaptureDependencies 706
#define RPC_cudaEventCreate 707
#define RPC_cudaEventCreateWithFlags 708
#define RPC_cudaEventRecord 709
#define RPC_cudaEventRecordWithFlags 710
#define RPC_cudaEventQuery 711
#define RPC_cudaEventSynchronize 712
#define RPC_cudaEventDestroy 713
#define RPC_cudaEventElapsedTime 714
#define RPC_cudaImportExternalMemory 715
#define RPC_cudaExternalMemoryGetMappedBuffer 716
#define RPC_cudaExternalMemoryGetMappedMipmappedArray 717
#define RPC_cudaDestroyExternalMemory 718
#define RPC_cudaImportExternalSemaphore 719
#define RPC_cudaSignalExternalSemaphoresAsync_v2 720
#define RPC_cudaWaitExternalSemaphoresAsync_v2 721
#define RPC_cudaDestroyExternalSemaphore 722
#define RPC_cudaLaunchKernel 723
#define RPC_cudaLaunchKernelExC 724
#define RPC_cudaLaunchCooperativeKernel 725
#define RPC_cudaLaunchCooperativeKernelMultiDevice 726
#define RPC_cudaFuncSetCacheConfig 727
#define RPC_cudaFuncSetSharedMemConfig 728
#define RPC_cudaFuncGetAttributes 729
#define RPC_cudaFuncSetAttribute 730
#define RPC_cudaSetDoubleForDevice 731
#define RPC_cudaSetDoubleForHost 732
#define RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor 733
#define RPC_cudaOccupancyAvailableDynamicSMemPerBlock 734
#define RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags 735
#define RPC_cudaOccupancyMaxPotentialClusterSize 736
#define RPC_cudaOccupancyMaxActiveClusters 737
#define RPC_cudaMallocManaged 738
#define RPC_cudaMalloc 739
#define RPC_cudaMallocHost 740
#define RPC_cudaMallocPitch 741
#define RPC_cudaMallocArray 742
#define RPC_cudaFree 743
#define RPC_cudaFreeHost 744
#define RPC_cudaFreeArray 745
#define RPC_cudaFreeMipmappedArray 746
#define RPC_cudaHostAlloc 747
#define RPC_cudaMalloc3D 748
#define RPC_cudaMalloc3DArray 749
#define RPC_cudaMallocMipmappedArray 750
#define RPC_cudaGetMipmappedArrayLevel 751
#define RPC_cudaMemcpy3D 752
#define RPC_cudaMemcpy3DPeer 753
#define RPC_cudaMemcpy3DAsync 754
#define RPC_cudaMemcpy3DPeerAsync 755
#define RPC_cudaMemGetInfo 756
#define RPC_cudaArrayGetInfo 757
#define RPC_cudaArrayGetPlane 758
#define RPC_cudaArrayGetMemoryRequirements 759
#define RPC_cudaMipmappedArrayGetMemoryRequirements 760
#define RPC_cudaArrayGetSparseProperties 761
#define RPC_cudaMipmappedArrayGetSparseProperties 762
#define RPC_cudaMemcpy 763
#define RPC_cudaMemcpy2DToArray 764
#define RPC_cudaMemcpy2DArrayToArray 765
#define RPC_cudaMemcpyToSymbol 766
#define RPC_cudaMemcpyAsync 767
#define RPC_cudaMemcpy2DToArrayAsync 768
#define RPC_cudaMemcpyToSymbolAsync 769
#define RPC_cudaMemset3D 770
#define RPC_cudaMemset3DAsync 771
#define RPC_cudaGetSymbolAddress 772
#define RPC_cudaGetSymbolSize 773
#define RPC_cudaMemPrefetchAsync 774
#define RPC_cudaMemAdvise 775
#define RPC_cudaMemRangeGetAttributes 776
#define RPC_cudaMemcpyToArray 777
#define RPC_cudaMemcpyArrayToArray 778
#define RPC_cudaMemcpyToArrayAsync 779
#define RPC_cudaMallocAsync 780
#define RPC_cudaMemPoolTrimTo 781
#define RPC_cudaMemPoolSetAccess 782
#define RPC_cudaMemPoolGetAccess 783
#define RPC_cudaMemPoolCreate 784
#define RPC_cudaMemPoolDestroy 785
#define RPC_cudaMallocFromPoolAsync 786
#define RPC_cudaMemPoolImportPointer 787
#define RPC_cudaPointerGetAttributes 788
#define RPC_cudaDeviceCanAccessPeer 789
#define RPC_cudaDeviceEnablePeerAccess 790
#define RPC_cudaDeviceDisablePeerAccess 791
#define RPC_cudaGraphicsUnregisterResource 792
#define RPC_cudaGraphicsResourceSetMapFlags 793
#define RPC_cudaGraphicsMapResources 794
#define RPC_cudaGraphicsUnmapResources 795
#define RPC_cudaGraphicsResourceGetMappedPointer 796
#define RPC_cudaGraphicsSubResourceGetMappedArray 797
#define RPC_cudaGraphicsResourceGetMappedMipmappedArray 798
#define RPC_cudaGetChannelDesc 799
#define RPC_cudaCreateChannelDesc 800
#define RPC_cudaCreateTextureObject 801
#define RPC_cudaDestroyTextureObject 802
#define RPC_cudaGetTextureObjectResourceDesc 803
#define RPC_cudaGetTextureObjectTextureDesc 804
#define RPC_cudaGetTextureObjectResourceViewDesc 805
#define RPC_cudaCreateSurfaceObject 806
#define RPC_cudaDestroySurfaceObject 807
#define RPC_cudaGetSurfaceObjectResourceDesc 808
#define RPC_cudaDriverGetVersion 809
#define RPC_cudaRuntimeGetVersion 810
#define RPC_cudaGraphCreate 811
#define RPC_cudaGraphAddKernelNode 812
#define RPC_cudaGraphKernelNodeGetParams 813
#define RPC_cudaGraphKernelNodeSetParams 814
#define RPC_cudaGraphKernelNodeCopyAttributes 815
#define RPC_cudaGraphKernelNodeGetAttribute 816
#define RPC_cudaGraphKernelNodeSetAttribute 817
#define RPC_cudaGraphAddMemcpyNode 818
#define RPC_cudaGraphAddMemcpyNodeToSymbol 819
#define RPC_cudaGraphMemcpyNodeGetParams 820
#define RPC_cudaGraphMemcpyNodeSetParams 821
#define RPC_cudaGraphMemcpyNodeSetParamsToSymbol 822
#define RPC_cudaGraphAddMemsetNode 823
#define RPC_cudaGraphMemsetNodeGetParams 824
#define RPC_cudaGraphMemsetNodeSetParams 825
#define RPC_cudaGraphAddHostNode 826
#define RPC_cudaGraphHostNodeGetParams 827
#define RPC_cudaGraphHostNodeSetParams 828
#define RPC_cudaGraphAddChildGraphNode 829
#define RPC_cudaGraphChildGraphNodeGetGraph 830
#define RPC_cudaGraphAddEmptyNode 831
#define RPC_cudaGraphAddEventRecordNode 832
#define RPC_cudaGraphEventRecordNodeGetEvent 833
#define RPC_cudaGraphEventRecordNodeSetEvent 834
#define RPC_cudaGraphAddEventWaitNode 835
#define RPC_cudaGraphEventWaitNodeGetEvent 836
#define RPC_cudaGraphEventWaitNodeSetEvent 837
#define RPC_cudaGraphAddExternalSemaphoresSignalNode 838
#define RPC_cudaGraphExternalSemaphoresSignalNodeGetParams 839
#define RPC_cudaGraphExternalSemaphoresSignalNodeSetParams 840
#define RPC_cudaGraphAddExternalSemaphoresWaitNode 841
#define RPC_cudaGraphExternalSemaphoresWaitNodeGetParams 842
#define RPC_cudaGraphExternalSemaphoresWaitNodeSetParams 843
#define RPC_cudaGraphAddMemAllocNode 844
#define RPC_cudaGraphMemAllocNodeGetParams 845
#define RPC_cudaDeviceGraphMemTrim 846
#define RPC_cudaGraphClone 847
#define RPC_cudaGraphNodeFindInClone 848
#define RPC_cudaGraphNodeGetType 849
#define RPC_cudaGraphGetNodes 850
#define RPC_cudaGraphGetRootNodes 851
#define RPC_cudaGraphGetEdges 852
#define RPC_cudaGraphNodeGetDependencies 853
#define RPC_cudaGraphNodeGetDependentNodes 854
#define RPC_cudaGraphAddDependencies 855
#define RPC_cudaGraphRemoveDependencies 856
#define RPC_cudaGraphDestroyNode 857
#define RPC_cudaGraphInstantiate 858
#define RPC_cudaGraphInstantiateWithFlags 859
#define RPC_cudaGraphInstantiateWithParams 860
#define RPC_cudaGraphExecGetFlags 861
#define RPC_cudaGraphExecKernelNodeSetParams 862
#define RPC_cudaGraphExecMemcpyNodeSetParams 863
#define RPC_cudaGraphExecMemcpyNodeSetParamsToSymbol 864
#define RPC_cudaGraphExecMemsetNodeSetParams 865
#define RPC_cudaGraphExecHostNodeSetParams 866
#define RPC_cudaGraphExecChildGraphNodeSetParams 867
#define RPC_cudaGraphExecEventRecordNodeSetEvent 868
#define RPC_cudaGraphExecEventWaitNodeSetEvent 869
#define RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams 870
#define RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams 871
#define RPC_cudaGraphNodeSetEnabled 872
#define RPC_cudaGraphNodeGetEnabled 873
#define RPC_cudaGraphExecUpdate 874
#define RPC_cudaGraphUpload 875
#define RPC_cudaGraphLaunch 876
#define RPC_cudaGraphExecDestroy 877
#define RPC_cudaGraphDestroy 878
#define RPC_cudaGraphDebugDotPrint 879
#define RPC_cudaUserObjectRetain 880
#define RPC_cudaUserObjectRelease 881
#define RPC_cudaGraphRetainUserObject 882
#define RPC_cudaGraphReleaseUserObject 883
#define RPC_cudaGetDriverEntryPoint 884
#define RPC_cudaGetExportTable 885
#define RPC_cudaGetFuncBySymbol 886
#define RPC_cublasCreate_v2 887
#define RPC_cublasDestroy_v2 888
#define RPC_cublasSgemm_v2 889
