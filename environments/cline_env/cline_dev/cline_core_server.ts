#!/usr/bin/env npx tsx

/**
 * Production Cline gRPC Server
 * 
 * Modified from test-standalone-core-api-server.ts to NOT start the ClineApiServerMock
 * which uses a hardcoded port 7777 that conflicts when running multiple workers.
 * 
 * This version uses the real Anthropic API instead of the mock.
 */

import * as fs from "node:fs"
import { mkdtempSync, rmSync } from "node:fs"
import * as os from "node:os"
import { ChildProcess, spawn } from "child_process"
import * as path from "path"

const PROTOBUS_PORT = process.env.PROTOBUS_PORT || "26040"
const HOSTBRIDGE_PORT = process.env.HOSTBRIDGE_PORT || "26041"
const WORKSPACE_DIR = process.env.WORKSPACE_DIR || process.cwd()
const E2E_TEST = process.env.E2E_TEST || "true"
const CLINE_ENVIRONMENT = process.env.CLINE_ENVIRONMENT || "local"
const USE_C8 = process.env.USE_C8 === "true"

// Locate the standalone build directory and core file
const projectRoot = process.env.PROJECT_ROOT || path.resolve(__dirname, "..", "cline")
const distDir = process.env.CLINE_DIST_DIR || path.join(projectRoot, "dist-standalone")
const clineCoreFile = process.env.CLINE_CORE_FILE || "cline-core.js"
const coreFile = path.join(distDir, clineCoreFile)

const childProcesses: ChildProcess[] = []

async function main(): Promise<void> {
	console.log("Starting Production Cline gRPC Server (no mock API)...")
	console.log(`Project Root: ${projectRoot}`)
	console.log(`Workspace: ${WORKSPACE_DIR}`)
	console.log(`ProtoBus Port: ${PROTOBUS_PORT}`)
	console.log(`HostBridge Port: ${HOSTBRIDGE_PORT}`)

	console.log(`Looking for standalone build at: ${coreFile}`)

	if (!fs.existsSync(coreFile)) {
		console.error(`Standalone build not found at: ${coreFile}`)
		console.error("To build the standalone version, run: npm run compile-standalone")
		process.exit(1)
	}

	// NOTE: We do NOT start ClineApiServerMock here - we use the real Anthropic API
	// This avoids the port 7777 conflict when running multiple workers

	const extensionsDir = path.join(distDir, "vsce-extension")
	const userDataDir = mkdtempSync(path.join(os.tmpdir(), "vsce"))
	const clineTestWorkspace = process.env.DEV_WORKSPACE_FOLDER || mkdtempSync(path.join(os.tmpdir(), "cline-test-workspace-"))

	console.log("Starting HostBridge test server...")
	const hostbridge: ChildProcess = spawn("npx", ["tsx", path.join(projectRoot, "scripts", "test-hostbridge-server.ts")], {
		stdio: "pipe",
		env: {
			...process.env,
			TEST_HOSTBRIDGE_WORKSPACE_DIR: clineTestWorkspace,
			HOST_BRIDGE_ADDRESS: `127.0.0.1:${HOSTBRIDGE_PORT}`,
		},
	})
	childProcesses.push(hostbridge)

	hostbridge.stdout?.on("data", (data) => {
		console.log(`[hostbridge] ${data.toString().trim()}`)
	})
	hostbridge.stderr?.on("data", (data) => {
		console.error(`[hostbridge] ${data.toString().trim()}`)
	})

	console.log(`Temp user data dir: ${userDataDir}`)
	console.log(`Temp extensions dir: ${extensionsDir}`)

	// Skip extraction - standalone.zip should be pre-extracted during build
	// Extracting at runtime causes race conditions when multiple workers run in parallel
	console.log("Using pre-extracted standalone (skipping runtime extraction)")

	// Wait for hostbridge to be ready
	await new Promise((resolve) => setTimeout(resolve, 1000))

	console.log("Starting Cline core...")
	const nodeArgs: string[] = USE_C8
		? ["npx", "c8", "--reporter=text", "--reporter=lcov", "node", coreFile]
		: ["node", coreFile]

	// CRITICAL: Cline core reads PROTOBUS_ADDRESS and HOST_BRIDGE_ADDRESS (full address),
	// NOT just the port numbers. Without these, it defaults to :26040 and :26041.
	const protobusAddress = `127.0.0.1:${PROTOBUS_PORT}`
	const hostbridgeAddress = `127.0.0.1:${HOSTBRIDGE_PORT}`
	console.log(`Configuring Cline core: PROTOBUS_ADDRESS=${protobusAddress}, HOST_BRIDGE_ADDRESS=${hostbridgeAddress}`)

	const clineProcess = spawn(nodeArgs[0], nodeArgs.slice(1), {
		stdio: "inherit",
		env: {
			...process.env,
			// Must use _ADDRESS vars, not just _PORT - Cline reads these
			PROTOBUS_ADDRESS: protobusAddress,
			HOST_BRIDGE_ADDRESS: hostbridgeAddress,
			// Keep port vars for compatibility
			PROTOBUS_PORT,
			HOSTBRIDGE_PORT,
			WORKSPACE_DIR: clineTestWorkspace,
			DEV_WORKSPACE_FOLDER: clineTestWorkspace,
			E2E_TEST,
			CLINE_ENVIRONMENT,
			VSCODE_USER_DATA_DIR: userDataDir,
			VSCODE_EXTENSIONS_DIR: extensionsDir,
		},
	})
	childProcesses.push(clineProcess)

	clineProcess.on("exit", (code) => {
		console.log(`Cline core exited with code ${code}`)
		cleanup()
		process.exit(code || 0)
	})

	process.on("SIGINT", cleanup)
	process.on("SIGTERM", cleanup)
}

function cleanup(): void {
	console.log("Cleaning up child processes...")
	for (const proc of childProcesses) {
		if (proc && !proc.killed) {
			proc.kill()
		}
	}
}

main().catch((error) => {
	console.error("Fatal error:", error)
	cleanup()
	process.exit(1)
})
