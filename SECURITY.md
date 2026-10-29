# Security policy

## Supported versions

Security fixes are applied to the current `v0.3.x` line.

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** for this repository:
open the **Security** tab and choose **Report a vulnerability**. Do not open a
public issue for a sensitive report.

Include the affected version/commit, a minimal reproduction, and the impact.
Model weights are not distributed in this repository; download them from the
upstream LiquidAI release and verify their checksums there.

## Scope

The engine, Metal shaders, HTTP server and build files are in scope. Third-party
headers under `third_party/metal-cpp` retain their upstream license and security
process.
