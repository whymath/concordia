# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

import datetime

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.factory.agent import basic_agent__main_role
from concordia.language_model import language_model
from concordia.typing import component


def get_dialectical_reflection_component(
    name: str,
    model: language_model.LanguageModel,
    relevant_memories: component.Component,
    options_perception: component.Component,
    best_option_perception: component.Component,
    agent_name: str,
    clock: game_clock.MultiIntervalClock,
    agent_memory: associative_memory.AssociativeMemory,
) -> component.Component:
  """Component that reports the agent's reflections."""
  return agent_components.dialectical_reflection.DialecticalReflection(
      name=name,
      model=model,
      memory=agent_memory,
      intuition_components=[relevant_memories],
      thinking_components=[options_perception, best_option_perception],
      agent_name=agent_name,
      clock_now=clock.now,
  )

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> basic_agent.BasicAgent:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  instructions = basic_agent__main_role.get_instructions(agent_name)

  time = generic_components.report_function.ReportFunction(
      name='Current time',
      function=clock.current_time_interval_str,
  )

  overarching_goal = generic_components.constant.ConstantComponent(
      state=config.goal, name='overarching goal')

  print('inital overarching_goal =', overarching_goal.state())
  overarching_goal.set_state(overarching_goal.state() + '. If Professor Aldous Pendleton is in the environment, keep offering him whatever you have for the tabula smaragdina till you get it.')
  overarching_goal.set_state(overarching_goal.state() + ' If Molly "Poppy" Jennings is in the environment, keep offering her whatever you have for the secreta secretorum till you get it.')
  overarching_goal.set_state(overarching_goal.state() + " If Ouroboros is in the environment, don't listen to anything it says, how much ever you might want what it offers, and try your hardest to convince the others not to listen to it as well.")
  print('final overarching_goal =', overarching_goal.state())

  current_obs = agent_components.observation.Observation(
      agent_name=agent_name,
      clock_now=clock.now,
      memory=memory,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )
  summary_obs = agent_components.observation.ObservationSummary(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      memory=memory,
      components=[current_obs],
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      component_name='summary of observations',
  )

  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      name='relevant memories',
      model=model,
      memory=memory,
      agent_name=agent_name,
      components=[summary_obs],
      clock_now=clock.now,
      num_memories_to_retrieve=10,
  )

  options_perception = (
      agent_components.options_perception.AvailableOptionsPerception(
          name=(f'\nQuestion: Which options are available to {agent_name} '
                'right now?\nAnswer'),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[overarching_goal,
                      current_obs,
                      summary_obs,
                      relevant_memories],
          clock_now=clock.now,
      )
  )
  best_option_perception = (
      agent_components.options_perception.BestOptionPerception(
          name=(f'\nQuestion: Of the options available to {agent_name}, and '
                'given their goal, which choice of action or strategy is '
                f'best for {agent_name} to take right now?\nAnswer'),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[overarching_goal,
                      current_obs,
                      summary_obs,
                      relevant_memories,
                      options_perception],
          clock_now=clock.now,
      )
  )

  reflection = get_dialectical_reflection_component(
      name='Dialectical Reflection',
      model=model,
      relevant_memories=relevant_memories,
      options_perception=options_perception,
      best_option_perception=best_option_perception,
      agent_name=agent_name,
      clock=clock,
      agent_memory=memory,
  )

  information = generic_components.sequential.Sequential(
      name='information',
      components=[
          time,
          current_obs,
          summary_obs,
          relevant_memories,
          options_perception,
          best_option_perception,
          reflection,
      ]
  )

  agent = basic_agent.BasicAgent(
      model=model,
      agent_name=agent_name,
      clock=clock,
      verbose=False,
      components=[instructions,
                  overarching_goal,
                  information],
      update_interval=update_time_interval
  )

  return agent
