# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Literal, Optional, Union, Any
from google import genai
from google.genai import types
from google.genai.types import (
    Part,
    GenerateContentConfig,
    Content,
    Candidate,
    FunctionResponse,
    FinishReason,
)
import time
import json_logger
from pathlib import Path

from computers import EnvState, Computer

MAX_RECENT_TURN_WITH_SCREENSHOTS = 3
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]


logger = json_logger.get_json_logger(__name__)

# Built-in Computer Use tools will return "EnvState".
# Custom provided functions will return "dict".
FunctionResponseT = Union[EnvState, dict]


def multiply_numbers(x: float, y: float) -> dict:
    """Multiplies two numbers."""
    return {"result": x * y}


class BrowserAgent:
    def __init__(
        self,
        browser_computer: Computer,
        query: str,
        model_name: str,
        verbose: bool = True,
    ):
        self._browser_computer = browser_computer
        self._query = query
        self._model_name = model_name
        self._verbose = verbose
        self.final_reasoning = None
        self._iteration_count = 0
        self._screenshot_count = 0
        
        # Set up screenshot directory
        self._screenshot_dir = Path(f"screenshots/run_{int(time.time())}")
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Screenshot directory created",
            extra={"extra_fields": {"screenshot_dir": str(self._screenshot_dir)}}
        )
        
        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            vertexai=os.environ.get("USE_VERTEXAI", "0").lower() in ["true", "1"],
            project=os.environ.get("VERTEXAI_PROJECT"),
            location=os.environ.get("VERTEXAI_LOCATION"),
        )
        self._contents: list[Content] = [
            Content(
                role="user",
                parts=[
                    Part(text=self._query),
                ],
            )
        ]

        # Exclude any predefined functions here.
        excluded_predefined_functions = []

        # Add your own custom functions here.
        custom_functions = [
            # For example:
            types.FunctionDeclaration.from_callable(
                client=self._client, callable=multiply_numbers
            )
        ]

        self._generate_content_config = GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=excluded_predefined_functions,
                    ),
                ),
                types.Tool(function_declarations=custom_functions),
            ],
        )

    def handle_action(self, action: types.FunctionCall) -> FunctionResponseT:
        """Handles the action and returns the environment state."""
        if action.name == "open_web_browser":
            return self._browser_computer.open_web_browser()
        elif action.name == "click_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.click_at(
                x=x,
                y=y,
            )
        elif action.name == "hover_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.hover_at(
                x=x,
                y=y,
            )
        elif action.name == "type_text_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            press_enter = action.args.get("press_enter", False)
            clear_before_typing = action.args.get("clear_before_typing", True)
            return self._browser_computer.type_text_at(
                x=x,
                y=y,
                text=action.args["text"],
                press_enter=press_enter,
                clear_before_typing=clear_before_typing,
            )
        elif action.name == "scroll_document":
            return self._browser_computer.scroll_document(action.args["direction"])
        elif action.name == "scroll_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            magnitude = action.args.get("magnitude", 800)
            direction = action.args["direction"]

            if direction in ("up", "down"):
                magnitude = self.denormalize_y(magnitude)
            elif direction in ("left", "right"):
                magnitude = self.denormalize_x(magnitude)
            else:
                raise ValueError("Unknown direction: ", direction)
            return self._browser_computer.scroll_at(
                x=x, y=y, direction=direction, magnitude=magnitude
            )
        elif action.name == "wait_5_seconds":
            return self._browser_computer.wait_5_seconds()
        elif action.name == "go_back":
            return self._browser_computer.go_back()
        elif action.name == "go_forward":
            return self._browser_computer.go_forward()
        elif action.name == "search":
            return self._browser_computer.search()
        elif action.name == "navigate":
            return self._browser_computer.navigate(action.args["url"])
        elif action.name == "key_combination":
            return self._browser_computer.key_combination(
                action.args["keys"].split("+")
            )
        elif action.name == "drag_and_drop":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            destination_x = self.denormalize_x(action.args["destination_x"])
            destination_y = self.denormalize_y(action.args["destination_y"])
            return self._browser_computer.drag_and_drop(
                x=x,
                y=y,
                destination_x=destination_x,
                destination_y=destination_y,
            )
        # Handle the custom function declarations here.
        elif action.name == multiply_numbers.__name__:
            return multiply_numbers(x=action.args["x"], y=action.args["y"])
        else:
            raise ValueError(f"Unsupported function: {action}")

    def get_model_response(
        self, max_retries=5, base_delay_s=1
    ) -> types.GenerateContentResponse:
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=self._contents,
                    config=self._generate_content_config,
                )
                return response  # Return response on success
            except Exception as e:
                logger.error("Error generating content", extra={"extra_fields": {"error": str(e), "attempt": attempt + 1}})
                if attempt < max_retries - 1:
                    delay = base_delay_s * (2**attempt)
                    logger.warning(
                        "Generating content failed, retrying",
                        extra={"extra_fields": {"attempt": attempt + 1, "max_retries": max_retries, "retry_delay_seconds": delay}}
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Generating content failed after all attempts",
                        extra={"extra_fields": {"max_retries": max_retries}}
                    )
                    raise

    def get_text(self, candidate: Candidate) -> Optional[str]:
        """Extracts the text from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return None
        text = []
        for part in candidate.content.parts:
            if part.text:
                text.append(part.text)
        return " ".join(text) or None

    def extract_function_calls(self, candidate: Candidate) -> list[types.FunctionCall]:
        """Extracts the function call from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return []
        ret = []
        for part in candidate.content.parts:
            if part.function_call:
                ret.append(part.function_call)
        return ret

    def run_one_iteration(self) -> Literal["COMPLETE", "CONTINUE"]:
        self._iteration_count += 1
        # Generate a response from the model.
        try:
            response = self.get_model_response()
        except Exception as e:
            return "COMPLETE"

        if not response.candidates:
            logger.error(
                "Response has no candidates",
                extra={"extra_fields": {"response": str(response)}}
            )
            raise ValueError("Empty response")

        # Extract the text and function call from the response.
        candidate = response.candidates[0]
        # Append the model turn to conversation history.
        if candidate.content:
            self._contents.append(candidate.content)

        reasoning = self.get_text(candidate)
        function_calls = self.extract_function_calls(candidate)

        # Retry the request in case of malformed FCs.
        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"

        if not function_calls:
            logger.info(
                "Agent Loop Complete",
                extra={"extra_fields": {"reasoning": reasoning}}
            )
            self.final_reasoning = reasoning
            return "COMPLETE"

        function_calls_data = []
        for function_call in function_calls:
            function_call_data = {"name": function_call.name}
            if function_call.args:
                function_call_data["args"] = function_call.args
            function_calls_data.append(function_call_data)
        
        logger.info(
            "Model response received",
            extra={"extra_fields": {"reasoning": reasoning, "function_calls": function_calls_data}}
        )

        function_responses = []
        for function_call in function_calls:
            extra_fr_fields = {}
            if function_call.args and (
                safety := function_call.args.get("safety_decision")
            ):
                decision = self._get_safety_confirmation(safety)
                if decision == "TERMINATE":
                    logger.info("Terminating agent loop", extra={"extra_fields": {"reason": "safety_confirmation"}})
                    return "COMPLETE"
                # Explicitly mark the safety check as acknowledged.
                extra_fr_fields["safety_acknowledgement"] = "true"
            logger.debug(
                "Executing function call",
                extra={"extra_fields": {"function_name": function_call.name, "function_args": function_call.args}}
            )
            fc_result = self.handle_action(function_call)
            if isinstance(fc_result, EnvState):
                # Save screenshot to disk
                screenshot_path = self._save_screenshot(
                    fc_result.screenshot,
                    function_call.name,
                    fc_result.url
                )
                
                function_responses.append(
                    FunctionResponse(
                        name=function_call.name,
                        response={
                            "url": fc_result.url,
                            "screenshot_path": screenshot_path,
                            **extra_fr_fields,
                        },
                        parts=[
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png", data=fc_result.screenshot
                                )
                            )
                        ],
                    )
                )
            elif isinstance(fc_result, dict):
                function_responses.append(
                    FunctionResponse(name=function_call.name, response=fc_result)
                )

        self._contents.append(
            Content(
                role="user",
                parts=[Part(function_response=fr) for fr in function_responses],
            )
        )

        # only keep screenshots in the few most recent turns, remove the screenshot images from the old turns.
        turn_with_screenshots_found = 0
        for content in reversed(self._contents):
            if content.role == "user" and content.parts:
                # check if content has screenshot of the predefined computer use functions.
                has_screenshot = False
                for part in content.parts:
                    if (
                        part.function_response
                        and part.function_response.parts
                        and part.function_response.name
                        in PREDEFINED_COMPUTER_USE_FUNCTIONS
                    ):
                        has_screenshot = True
                        break

                if has_screenshot:
                    turn_with_screenshots_found += 1
                    # remove the screenshot image if the number of screenshots exceed the limit.
                    if turn_with_screenshots_found > MAX_RECENT_TURN_WITH_SCREENSHOTS:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name
                                in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                part.function_response.parts = None

        return "CONTINUE"

    def _get_safety_confirmation(
        self, safety: dict[str, Any]
    ) -> Literal["CONTINUE", "TERMINATE"]:
        if safety["decision"] != "require_confirmation":
            raise ValueError(f"Unknown safety decision: safety['decision']")
        logger.warning(
            "Safety service requires explicit confirmation",
            extra={"extra_fields": {"explanation": safety.get("explanation", "")}}
        )
        decision = ""
        while decision.lower() not in ("y", "n", "ye", "yes", "no"):
            decision = input("Do you wish to proceed? [Yes]/[No]\n")
        logger.info(
            "Safety confirmation received",
            extra={"extra_fields": {"decision": decision}}
        )
        if decision.lower() in ("n", "no"):
            return "TERMINATE"
        return "CONTINUE"

    def agent_loop(self):
        status = "CONTINUE"
        while status == "CONTINUE":
            status = self.run_one_iteration()

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._browser_computer.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._browser_computer.screen_size()[1])
    
    def _save_screenshot(self, screenshot: bytes, action_name: str, url: str) -> str:
        """Save a screenshot to disk and return the file path."""
        self._screenshot_count += 1
        timestamp = int(time.time() * 1000)  # milliseconds for unique filenames
        # Sanitize action name for filename
        safe_action_name = action_name.replace("_", "-")
        filename = f"{self._screenshot_count:04d}_{timestamp}_{safe_action_name}.png"
        filepath = self._screenshot_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(screenshot)
        
        logger.debug(
            "Screenshot saved",
            extra={"extra_fields": {"filepath": str(filepath), "action": action_name, "url": url}}
        )
        
        return str(filepath)
