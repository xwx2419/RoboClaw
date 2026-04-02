MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LOCAL_PKG_DIR ?= $(MAKEFILE_DIR)/.a2d_pkg
COROBOT_WHL ?=
COROBOT_SITE_PACKAGES ?=
ROOT_PYTHON ?= 3.10
MCP_PYTHON ?= 3.12
BASIC_MEMORY_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/basic-memory
METASEARCH_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/metasearch-mcp
COROBOT_MCP_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/corobot_mcp_server
DATA_ANALYST_MCP_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/data_analyst_mcp_server

# 默认仅注入 src；本地扩展目录存在时自动加入；可选追加授权用户的 site-packages
PYTHONPATH_VALUE := $(MAKEFILE_DIR)/src$(if $(wildcard $(LOCAL_PKG_DIR)),:$(LOCAL_PKG_DIR),)$(if $(COROBOT_SITE_PACKAGES),:$(COROBOT_SITE_PACKAGES),)
PYENV := PYTHONPATH=$(PYTHONPATH_VALUE)
LD_LIBRARY := LD_LIBRARY_PATH=/data/opencv45:$LD_LIBRARY_PATH
CURRENT_TIME := $(shell date +"%Y-%m-%d_%H-%M")
UV_RUN_ROOT := uv run --python $(ROOT_PYTHON)


init:
	mkdir -p ./applog/
	git submodule update --init --recursive
	uv python install $(ROOT_PYTHON) $(MCP_PYTHON)
	uv sync --frozen --python $(ROOT_PYTHON)
	$(UV_RUN_ROOT) pre-commit install
	uv --directory "$(BASIC_MEMORY_DIR)" sync --frozen --python $(MCP_PYTHON)
	uv --directory "$(METASEARCH_DIR)" sync --frozen --python $(MCP_PYTHON)
	uv --directory "$(COROBOT_MCP_DIR)" sync --python $(ROOT_PYTHON)
	uv --directory "$(DATA_ANALYST_MCP_DIR)" sync --python $(ROOT_PYTHON)

install_g01_whl:
	@test -n "$(COROBOT_WHL)" || (echo "请提供 COROBOT_WHL=/path/to/corobot-*.whl" && exit 1)
	@test -f "$(COROBOT_WHL)" || (echo "未找到 whl 文件: $(COROBOT_WHL)" && exit 1)
	mkdir -p "$(LOCAL_PKG_DIR)"
	uv pip install --target "$(LOCAL_PKG_DIR)" "$(COROBOT_WHL)"


test:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/llm_manager/openai_client/test_openai_client.py

run:
	$(MAKE) run_tui

run_a2d:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/cmd/olympus_img_cmd.py

run_gui:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/gradio_ui/gradio_ui.py

run_tui:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/tui/olympus_tui.py

test_img:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/llm_manager/openai_client/test_img.py

test_mm:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/memory_manager/test_memory_manager.py

test_a2d:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/machine_layer/test_dataloader_a2d.py

test_udp:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/interaction_layer/test_udp.py

test_a2d_img:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/test_img_agent.py
