#include "oracle_solver.h"

#include "oracle_response_parser.h"
#include "../expr2sygus.h"
#include "delphi/synthesis/cvc4_synth.h"

#include <util/expr.h>
#include <util/format_expr.h>
#include <util/run.h>
#include <util/std_expr.h>
#include <util/string_utils.h>

#include <sstream>
#include <iostream>

#include <solvers/smt2/smt2_dec.h>
#include <util/tempfile.h>

oracle_solvert::oracle_solvert(
  decision_proceduret &__sub_solver,
  repr_syntht __repr_type,
  message_handlert &__message_handler) :
  sub_solver(__sub_solver),
  oracle_repr_type(__repr_type),
  log(__message_handler)
{
}

exprt oracle_solvert::get_oracle_value(const function_application_exprt &oracle_app)
{
  for (const auto &app : applications)
  {
    if(to_function_application_expr(app.first) == oracle_app)
    {
      // get inputs
      std::vector<exprt> inputs;
      inputs.reserve(app.second.argument_handles.size());

      for (auto &argument_handle : app.second.argument_handles)
      {
        auto res = get(argument_handle);
        inputs.push_back(res);
      }
      auto history = oracle_call_history.find(app.second.binary_name);
      INVARIANT(history != oracle_call_history.end(), "No history for oracle");
      auto result = history->second.find(inputs);
      if (result == history->second.end())
      {
        return call_oracle(app.second, inputs);
      }
      return result->second;
    }
  }
  return nil_exprt();
}

void oracle_solvert::set_to(const exprt &expr, bool value)
{
  PRECONDITION(oracle_fun_map != nullptr);

  // find any oracle function applications
  expr.visit_pre([this](const exprt &src) {
    if(src.id() == ID_function_application)
    {
      auto &application_expr = to_function_application_expr(src);
      if(application_expr.function().id() == ID_symbol)
      {
        // look up whether it is an oracle
        auto identifier = to_symbol_expr(application_expr.function()).get_identifier();
        auto oracle_fun_map_it = oracle_fun_map->find(identifier);
        if(oracle_fun_map_it != oracle_fun_map->end())
        {
          // yes
          if(applications.find(application_expr) == applications.end())
          {
            // std::cout<<"adding a new application "<< expr2sygus(application_expr)<<std::endl;
            // not seen before
            auto &application = applications[application_expr];
            application.binary_name = oracle_fun_map_it->second.binary_name;
            application.handle = handle(application_expr);

            application.argument_handles.reserve(application_expr.arguments().size());

            for(auto &argument : application_expr.arguments())
              application.argument_handles.push_back(handle(argument));
          }
        }
      }
    }
  });

  sub_solver.set_to(expr, value);
}

exprt oracle_solvert::handle(const exprt &expr)
{
  if(expr.is_constant())
    return expr;
  else
  {
    symbol_exprt symbol_expr("H"+std::to_string(handle_counter++), expr.type());
    auto equality = equal_exprt(symbol_expr, expr);
    set_to_true(equality);
    return std::move(symbol_expr);
  }
}

exprt oracle_solvert::get(const exprt &expr) const
{
  return sub_solver.get(expr);
}

void oracle_solvert::print_assignment(std::ostream &out) const
{
  sub_solver.print_assignment(out);
}

std::string oracle_solvert::decision_procedure_text() const
{
  return "oracles + " + sub_solver.decision_procedure_text();
}

oracle_solvert::check_resultt oracle_solvert::check_oracles()
{
  oracle_solvert::check_resultt result = CONSISTENT;

// std::cout<<"There are "<< applications.size()<<" applications in check oracles\n";
  for(const auto &application : applications)
  { 
    switch(check_oracle(application.first, application.second))
    {
    case INCONSISTENT:
      result = INCONSISTENT;
      break;

    case CONSISTENT:
      break;

    case ERROR:
      return ERROR; // abort
    }
  }

  return result;
}

exprt oracle_solvert::make_oracle_call(const std::string &binary_name, const std::vector<std::string> &argv)
{
  log.debug() << "Running oracle (verification) ";
  for (auto &arg : argv)
    log.debug() << ' ' << arg;
  log.debug() << messaget::eom;

  // run the oracle binary
  std::ostringstream stdout_stream;

  auto run_result = run(
      binary_name,
      argv,
      "",
      stdout_stream,
      "");

  if (run_result != 0 && run_result !=10)
  {
    log.error() << "oracle " << binary_name << " has failed with exit code " << integer2string(run_result) << messaget::eom;
    // assert(0);
    // return nil_exprt();
  }
  // we assume that the oracle returns the result in SMT-LIB format
  std::istringstream oracle_response_istream(stdout_stream.str());
  log.debug() << "Oracle response is "<< stdout_stream.str() << messaget::eom;
  return oracle_response_parser(oracle_response_istream);
}

exprt oracle_solvert::call_oracle(
    const applicationt &application, const std::vector<exprt> &inputs)
    {
      bool is_new = false;
      return call_oracle(application, inputs, is_new);
    }

exprt oracle_solvert::call_oracle(
    const applicationt &application, const std::vector<exprt> &inputs, bool &is_new_call)
{
  if(cache && oracle_call_history.find(application.binary_name) == oracle_call_history.end())
  {
    oracle_call_history[application.binary_name] = oracle_historyt();
  }

  exprt response;
  if (oracle_call_history[application.binary_name].find(inputs) == oracle_call_history[application.binary_name].end() || !cache)
  {
    is_new_call = true;
    std::vector<std::string> argv;
    argv.push_back(application.binary_name);

    for (const auto &input : inputs)
    {
      std::ostringstream stream;
      stream << format(input);
      argv.push_back(stream.str());
    }

    response = make_oracle_call(application.binary_name, argv);
    if (cache)
      oracle_call_history[application.binary_name][inputs] = response;
  }
  else
  {
    is_new_call = false;
    response = oracle_call_history[application.binary_name][inputs];
  }

  return response;
}

oracle_solvert::check_resultt oracle_solvert::check_oracle(
  const function_application_exprt &application_expr,
  const applicationt &application)
{
  
  // evaluate the argument handles to get concrete inputs
  std::vector<exprt> inputs;
  inputs.reserve(application.argument_handles.size());

  for(auto &argument_handle : application.argument_handles)
  {
    auto res = get(argument_handle);
    inputs.push_back(res);
  } 

   exprt response = call_oracle(application, inputs);
   if(response==nil_exprt())
     return check_resultt::ERROR;


  // check whether the result is consistent with the model
  if(response == get(application.handle))
  {
    // log.debug() << "Response matches " << expr2sygus(get(application.handle))<<messaget::eom;
    return CONSISTENT; // done, SAT
  }
  {

    function_application_exprt func_app(application_expr.function(), inputs);

    // log.debug() << "Response does not match " << expr2sygus(get(application.handle)) << messaget::eom;

    // add a constraint that enforces this equality
    auto response_equality = equal_exprt(application.handle, response);
    // auto response_equality = equal_exprt(func_app, response);
    // set_to_true(response_equality);

    exprt::operandst input_constraints;

    for (auto &argument_handle : application.argument_handles)
      input_constraints.push_back(equal_exprt(argument_handle, get(argument_handle)));

    // add 'all inputs equal' => 'return value equal'
    auto implication =
        implies_exprt(
            conjunction(input_constraints),
            response_equality);
    sub_solver.set_to_true(implication);        

  }
  return INCONSISTENT;
}

void oracle_solvert::synth_oracle_representations() {
    console_message_handlert message_handler;
    messaget message(message_handler);
    cvc4_syntht synthesizer(message_handler, true, false, true, false);

    for (const auto &history : oracle_call_history) {
        /* if (history.second.size() < 3) { */
        /*     continue; */
        /* } */

        problemt repr_problem;

        irep_idt repr_name = history.first + "_repr";
        for (const auto &funmappair : *oracle_fun_map) {
            if (funmappair.second.binary_name == history.first) {
                const auto &oracle_fun = funmappair.second;
                const size_t num_params = oracle_fun.type.domain().size();

                std::vector<irep_idt> repr_params;
                for (size_t i = 0; i < num_params; ++i)
                    repr_params.emplace_back("p" + integer2string(i) + "#" + integer2string(i));

                typet repr_ret_type = oracle_fun.type.codomain();

                synth_functiont repr_funt(oracle_fun.type);
                repr_funt.parameters = repr_params;
                repr_problem.synthesis_functions.insert(std::make_pair(repr_name, repr_funt));

                std::cout << "[ORACLE REPR] COMPOSING PROBLEM" << std::endl;
                std::stringstream params_formatted;
                for (const auto &call : history.second) {

                    symbol_exprt repr_func_symbol = symbol_exprt(repr_name, oracle_fun.type);
                    function_application_exprt repr_func_appl = function_application_exprt(repr_func_symbol, call.first);
                    exprt repr_ex_equality = equal_exprt(repr_func_appl, call.second);
                    repr_problem.synthesis_constraints.insert(repr_ex_equality);

                    for (const auto &call_arg : call.first) {
                        params_formatted << expr2sygus(call_arg) << ",";
                    }
                    params_formatted << " | ";
                    params_formatted << expr2sygus(call.second);
                    params_formatted << "\n";
                }

                std::cout << "[ORACLE REPR] SOLVING" << std::endl;
                decision_proceduret::resultt result = synthesizer.solve(repr_problem);
                switch (result)
                {
                    case decision_proceduret::resultt::D_SATISFIABLE:
                        std::cout << "[ORACLE REPR] CANDIDATE FOUND:" << std::endl;
                        for (const auto &sol : synthesizer.get_solution().functions) {
                            std::cout << "lhs: " << expr2sygus(sol.first) << std::endl;
                            std::cout << "rhs: " << expr2sygus(sol.second) << std::endl;

                            // should only be one function
                            oracle_representations[history.first] = expr2sygus(sol.second);
                        }
                        break;

                    case decision_proceduret::resultt::D_UNSATISFIABLE:
                        std::cout << "[ORACLE REPR] UNSAT:" << std::endl;
                        break;

                    case decision_proceduret::resultt::D_ERROR:
                    default:
                        std::cout << "[ORACLE REPR] ERROR:" << std::endl;
                        break;
                }

            }
        }
    }

}

void oracle_solvert::learn_oracle_representations() {
    console_message_handlert message_handler;
    messaget message(message_handler);

    for (const auto &history : oracle_call_history) {
        /* if (history.second.size() < 3) { */
        /*     continue; */
        /* } */
        for (const auto &funmappair : *oracle_fun_map) {
            if (funmappair.second.binary_name == history.first) {
                const auto &oracle_fun = funmappair.second;
                const size_t num_params = oracle_fun.type.domain().size();

                std::vector<irep_idt> repr_params;
                for (size_t i = 0; i < num_params; ++i)
                    repr_params.push_back("p" + integer2string(i) + "#" + integer2string(i));
                typet repr_ret_type = oracle_fun.type.codomain();

                std::stringstream params_formatted;
                for (const auto &call : history.second) {
                    for (const auto &call_arg : call.first) {
                        params_formatted << expr2sygus(call_arg) << ",";
                    }
                    params_formatted << " | ";
                    params_formatted << expr2sygus(call.second);
                    params_formatted << "\n";
                }

                temporary_filet
                        temp_file_problem("nn_problem_", ""),
                        temp_file_stdout("nn_stdout_", ""),
                        temp_file_stderr("nn_stderr_", "");
                {
                    std::ofstream problem_out(
                            temp_file_problem(), std::ios_base::out | std::ios_base::trunc);
                    problem_out << params_formatted.str();
                }


                std::vector<std::string> argv;
                std::string stdin_filename;

                argv = {"python", "/Users/apdoshi/syn-delphi/representation/run_logics_nn.py", temp_file_problem()};

                std::cout << "[ORACLE REPR] RUNNING: " << argv[0] << " " << argv[1] << " " << argv[2] << std::endl;

                int res =
                        run(argv[0], argv, stdin_filename, temp_file_stdout(), temp_file_stderr());

                if (res < 0) {
                    std::cout << "[ORACLE REPR] ERROR:" << std::endl;
                    break;
                } else
                {
                    std::ifstream in(temp_file_stdout());
                    std::stringstream buffer;
                    buffer << in.rdbuf();
                    std::string result = buffer.str();
                    std::cout << "[ORACLE REPR] RES:" << std::endl;
                    std::cout << result << std::endl;
                    if (!result.empty()) {
                        oracle_representations[history.first] = result;
                    }
                }
            }
        }
    }

}

void oracle_solvert::substitute_oracles() {
  std::unordered_map<std::string, std::string> name2funcdefinition;
  for (const auto& funmappair : *oracle_fun_map) {
    const std::string& smt2_identifier = id2string(funmappair.first);
    const std::string& binary_name = funmappair.second.binary_name;
    const auto& func_type = funmappair.second.type;

    // make sure oracle exists
    if (oracle_representations.find(binary_name) == oracle_representations.end()) continue;

    std::string new_fun = "(define-fun |" + smt2_identifier + "| (";
    // input arguments
    for (size_t i = 0; i < func_type.domain().size(); ++i) 
      new_fun += "(p" + integer2string(i) + " " + type2sygus(func_type.domain()[i]) + ") ";

    // output argument
    new_fun += ") " + type2sygus(func_type.codomain()) + " ";

    // function body
    new_fun += oracle_representations[binary_name] + ")\n";
    name2funcdefinition[smt2_identifier] = new_fun;
  }
  smt2_dect * cast_solver = dynamic_cast<smt2_dect *>(&sub_solver);
  cast_solver->substitute_oracles(name2funcdefinition);
}

decision_proceduret::resultt oracle_solvert::dec_solve()
{
  PRECONDITION(oracle_fun_map != nullptr);

  number_of_solver_calls++;

  while(true)
  {
      if (oracle_repr_type != NO_REPR) {
          if (oracle_repr_type == SYGUS_REPR) {
              // TODO maybe move sygus/calling cvc5 to Python as well
              synth_oracle_representations();
          } else {
              learn_oracle_representations();
          }
          substitute_oracles();
      }

    switch(sub_solver())
    {
    case resultt::D_SATISFIABLE:
      switch(check_oracles())
      {
      case INCONSISTENT:
        break; // constraint added, we'll do another iteration

      case CONSISTENT:
        return resultt::D_SATISFIABLE;

      case ERROR:
        return resultt::D_ERROR;
      }
      break;

    case resultt::D_UNSATISFIABLE:
      return resultt::D_UNSATISFIABLE;

    case resultt::D_ERROR:
      return resultt::D_ERROR;
    }
  }
}
