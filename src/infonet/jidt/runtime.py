"""Centralised JVM/JIDT bootstrapping."""
from __future__ import annotations

import os

import numpy as np

JIDT_CLASSPATH = os.environ.get(
    "INFONET_JIDT_JAR",
    "/home/r-env/jidt/infodynamics.jar",
)

JVM_MAX_HEAP = os.environ.get("INFONET_JVM_MAX_HEAP", "-Xmx4g")  # JVM heap


def ensure_jvm() -> None:
    """Start the JVM with the JIDT jar on the classpath, if not already running."""
    from jpype import isJVMStarted, startJVM, getDefaultJVMPath

    if not isJVMStarted():
        startJVM(
            getDefaultJVMPath(),
            "-ea",
            f"-Djava.class.path={JIDT_CLASSPATH}",
            JVM_MAX_HEAP,
            convertStrings=True,
        )


def to_jarray_1d(arr: np.ndarray) -> "JArray":
    """Convert a 1-D numpy array to a Java double[] for JIDT."""
    from jpype import JArray, JDouble

    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return JArray(JDouble, 1)(arr.tolist())


def to_jarray_2d(arr: np.ndarray) -> "JArray":
    """Convert an (n, k) numpy array to a Java double[][] for JIDT."""
    from jpype import JArray, JDouble

    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return JArray(JDouble, 2)(arr.tolist())