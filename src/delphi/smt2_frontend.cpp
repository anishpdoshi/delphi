/*******************************************************************\

Module: SMT2 Solver that uses boolbv and the default SAT solver

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <solvers/smt2/smt2_parser.h>
#include <solvers/smt2/smt2_format.h>
#include <solvers/smt2/smt2_dec.h>

#include <fstream>
#include <iostream>

#include <util/cmdline.h>
#include <util/message.h>
#include <util/namespace.h>
#include <util/replace_symbol.h>
#include <util/simplify_expr.h>
#include <util/symbol_table.h>

#include <solvers/sat/satcheck.h>
#include <solvers/flattening/boolbv.h>

#include "verification/oracle_solver.h"
#include "verification/negation_oracle_solver.h"

class smt2_solvert:public smt2_parsert
{
public:
  smt2_solvert(std::istream &_in, oracle_solvert &_solver)
    : smt2_parsert(_in), solver(_solver), status(NOT_SOLVED)
  {
    setup_commands();
  }

protected:
  oracle_solvert &solver;

  void setup_commands();
  void define_constants();
  void expand_function_applications(exprt &);

  std::set<irep_idt> constants_done;

  enum
  {
    NOT_SOLVED,
    SAT,
    UNSAT
  } status;
};

void smt2_solvert::define_constants()
{
  for(const auto &id : id_map)
  {
    if(id.second.type.id() == ID_mathematical_function)
      continue;

    if(id.second.definition.is_nil())
      continue;

    const irep_idt &identifier = id.first;

    // already done?
    if(constants_done.find(identifier)!=constants_done.end())
      continue;

    constants_done.insert(identifier);

    exprt def = id.second.definition;
    expand_function_applications(def);
    solver.set_to_true(
      equal_exprt(symbol_exprt(identifier, id.second.type), def));
  }
}

void smt2_solvert::expand_function_applications(exprt &expr)
{
  for(exprt &op : expr.operands())
    expand_function_applications(op);

  if(expr.id()==ID_function_application)
  {
    auto &app=to_function_application_expr(expr);

    if(app.function().id() == ID_symbol)
    {
      // look up the symbol
      auto identifier = to_symbol_expr(app.function()).get_identifier();
      auto f_it = id_map.find(identifier);

      if(f_it != id_map.end())
      {
        const auto &f = f_it->second;
        if(f.parameters.size()==0)
        {
          // this was UF
          return;
        }

        DATA_INVARIANT(
          f.type.id() == ID_mathematical_function,
          "type of function symbol must be mathematical_function_type");

        const auto &domain = to_mathematical_function_type(f.type).domain();

        DATA_INVARIANT(
          domain.size() == app.arguments().size(),
          "number of parameters must match number of arguments");

        replace_symbolt replace_symbol;

        for(std::size_t i = 0; i < domain.size(); i++)
        {
          replace_symbol.insert(
            symbol_exprt(f.parameters[i], domain[i]), app.arguments()[i]);
        }

        exprt body = f.definition;
        replace_symbol(body);
        expand_function_applications(body);
        expr = body;
      }
    }
  }
}

void smt2_solvert::setup_commands()
{
  {
    commands["assert"] = [this]() {
      exprt e = expression();
      if(e.is_not_nil())
      {
        solver.oracle_fun_map=&oracle_symbols;
        expand_function_applications(e);
        solver.set_to_true(e);
      }
    };

    commands["check-sat"] = [this]() {
      // add constant definitions as constraints
      solver.oracle_fun_map=&oracle_symbols;
      define_constants();

      switch(solver())
      {
      case decision_proceduret::resultt::D_SATISFIABLE:
        std::cout << "sat\n";
        status = SAT;
        break;

      case decision_proceduret::resultt::D_UNSATISFIABLE:
        std::cout << "unsat\n";
        status = UNSAT;
        break;

      case decision_proceduret::resultt::D_ERROR:
        std::cout << "error\n";
        status = NOT_SOLVED;
      }
    };

    commands["check-sat-assuming"] = [this]() {
      throw error("not yet implemented");
    };

    commands["display"] = [this]() {
      // this is a command that Z3 appears to implement
      exprt e = expression();
      if(e.is_not_nil())
        std::cout << smt2_format(e) << '\n';
    };

    commands["get-value"] = [this]() {
      std::vector<exprt> ops;

      if(next_token() != smt2_tokenizert::OPEN)
        throw error("get-value expects list as argument");

      while(smt2_tokenizer.peek() != smt2_tokenizert::CLOSE &&
            smt2_tokenizer.peek() != smt2_tokenizert::END_OF_FILE)
      {
        ops.push_back(expression()); // any term
      }

      if(next_token() != smt2_tokenizert::CLOSE)
        throw error("get-value expects ')' at end of list");

      if(status != SAT)
        throw error("model is not available");

      std::vector<exprt> values;
      values.reserve(ops.size());

      for(const auto &op : ops)
      {
        if(op.id() != ID_symbol)
          throw error("get-value expects symbol");

        const auto &identifier = to_symbol_expr(op).get_identifier();

        const auto id_map_it = id_map.find(identifier);

        if(id_map_it == id_map.end())
          throw error() << "unexpected symbol '" << identifier << '\'';

        const exprt value = solver.get(op);

        if(value.is_nil())
          throw error() << "no value for '" << identifier << '\'';

        values.push_back(value);
      }

      std::cout << '(';

      for(std::size_t op_nr = 0; op_nr < ops.size(); op_nr++)
      {
        if(op_nr != 0)
          std::cout << "\n ";

        std::cout << '(' << smt2_format(ops[op_nr]) << ' '
                  << smt2_format(values[op_nr]) << ')';
      }

      std::cout << ")\n";
    };

    commands["echo"] = [this]() {
      if(next_token() != smt2_tokenizert::STRING_LITERAL)
        throw error("expected string literal");

      std::cout << smt2_format(constant_exprt(
                     smt2_tokenizer.get_buffer(), string_typet()))
                << '\n';
    };

    commands["get-assignment"] = [this]() {
      // print satisfying assignment for all named expressions

      if(status != SAT)
        throw error("model is not available");

      bool first = true;

      std::cout << '(';
      for(const auto &named_term : named_terms)
      {
        const symbol_tablet symbol_table;
        const namespacet ns(symbol_table);
        const auto value =
          simplify_expr(solver.get(named_term.second.term), ns);

        if(value.is_constant())
        {
          if(first)
            first = false;
          else
            std::cout << '\n' << ' ';

          std::cout << '(' << smt2_format(named_term.second.name) << ' '
                    << smt2_format(value) << ')';
        }
      }
      std::cout << ')' << '\n';
    };

    commands["get-model"] = [this]() {
      // print a model for all identifiers

      if(status != SAT)
        throw error("model is not available");

      const symbol_tablet symbol_table;
      const namespacet ns(symbol_table);

      bool first = true;

      std::cout << "(\n";
      for(const auto &id : id_map)
      {
        const symbol_exprt name(id.first, id.second.type);
        const auto value = simplify_expr(solver.get(name), ns);

        if(value.is_not_nil())
        {
          if(first)
            first = false;
          else
            std::cout << '\n' << ' ';

          std::cout << "(define-fun " << smt2_format(name) << ' ';

          if(id.second.type.id() == ID_mathematical_function)
            throw error("models for functions unimplemented");
          else
            std::cout << "() " << smt2_format(id.second.type);

          std::cout << ' ' << smt2_format(value) << ')';
        }
      }
      std::cout << "\n)" << '\n';
    };

    commands["simplify"] = [this]() {
      // this is a command that Z3 appears to implement
      exprt e = expression();
      if(e.is_not_nil())
      {
        const symbol_tablet symbol_table;
        const namespacet ns(symbol_table);
        exprt e_simplified = simplify_expr(e, ns);
        std::cout << smt2_format(e) << '\n';
      }
    };
  }

#if 0
    // TODO:
    | ( declare-const hsymboli hsorti )
    | ( declare-datatype hsymboli hdatatype_deci)
    | ( declare-datatypes ( hsort_deci n+1 ) ( hdatatype_deci n+1 ) )
    | ( declare-fun hsymboli ( hsorti ??? ) hsorti )
    | ( declare-sort hsymboli hnumerali )
    | ( define-fun hfunction_def i )
    | ( define-fun-rec hfunction_def i )
    | ( define-funs-rec ( hfunction_deci n+1 ) ( htermi n+1 ) )
    | ( define-sort hsymboli ( hsymboli ??? ) hsorti )
    | ( get-assertions )
    | ( get-info hinfo_flag i )
    | ( get-option hkeywordi )
    | ( get-proof )
    | ( get-unsat-assumptions )
    | ( get-unsat-core )
    | ( pop hnumerali )
    | ( push hnumerali )
    | ( reset )
    | ( reset-assertions )
    | ( set-info hattributei )
    | ( set-option hoptioni )
#endif
}

class smt2_message_handlert : public message_handlert
{
public:
  void print(unsigned level, const std::string &message) override
  {
    message_handlert::print(level, message);

    if(level < 4) // errors
      std::cout << "(error \"" << message << "\")\n";
    else
      std::cout << "; " << message << '\n';
  }

  void print(unsigned, const xmlt &) override
  {
  }

  void print(unsigned, const jsont &) override
  {
  }

  void flush(unsigned) override
  {
    std::cout << std::flush;
  }
};


int negation_solver(std::istream &in, decision_proceduret &solver, decision_proceduret &solver2, oracle_solvert::oracle_repr_optionst repr_options, message_handlert &message_handler)
{
  messaget message(message_handler);

  negation_oracle_solvert oracle_solver(solver, solver2, repr_options, message_handler);
  smt2_solvert smt2_solver{in, oracle_solver};

  bool error_found = false;

  while(!smt2_solver.exit)
  {
    try
    {
      smt2_solver.parse();
    }
    catch(const smt2_tokenizert::smt2_errort &error)
    {
      smt2_solver.skip_to_end_of_list();
      error_found = true;
      message.error().source_location.set_line(error.get_line_no());
      message.error() << error.what() << messaget::eom;
    }
    catch(const analysis_exceptiont &error)
    {
      smt2_solver.skip_to_end_of_list();
      error_found = true;
      message.error() << error.what() << messaget::eom;
    }
  }

  if(error_found)
    return 20;
  else
    return 0;
}

int solver(std::istream &in, decision_proceduret &solver, negation_oracle_solvert::oracle_repr_optionst repr_options, message_handlert &message_handler)
{
  messaget message(message_handler);

  oracle_solvert oracle_solver(solver, std::move(repr_options), message_handler);
  smt2_solvert smt2_solver{in, oracle_solver};

  bool error_found = false;

  while(!smt2_solver.exit)
  {
    try
    {
      smt2_solver.parse();
    }
    catch(const smt2_tokenizert::smt2_errort &error)
    {
      smt2_solver.skip_to_end_of_list();
      error_found = true;
      message.error().source_location.set_line(error.get_line_no());
      message.error() << error.what() << messaget::eom;
    }
    catch(const analysis_exceptiont &error)
    {
      smt2_solver.skip_to_end_of_list();
      error_found = true;
      message.error() << error.what() << messaget::eom;
    }
  }

  if(error_found)
    return 20;
  else
    return 0;
}


int smt2_frontend(const cmdlinet &cmdline, std::istream &in)
{

  // this is our default verbosity
  unsigned int v=messaget::M_STATISTICS;

  if(cmdline.isset("verbosity"))
  {
    v=std::stol(
        cmdline.get_value("verbosity"));;
    if(v>10)
      v=10;
  }

  symbol_tablet symbol_table;
  namespacet ns(symbol_table);

  smt2_message_handlert message_handler;
  messaget message(message_handler);

  // this is our default verbosity
  message_handler.set_verbosity(messaget::M_STATISTICS);

  std::string logic="ALL";

  oracle_solvert::repr_syntht oracle_repr_type = oracle_solvert::repr_syntht::NO_REPR;
  if (cmdline.get_value("oracle-repr") == "sygus") {
    oracle_repr_type = oracle_solvert::repr_syntht::SYGUS_REPR;
  } else if (cmdline.get_value("oracle-repr") == "neural") {
    oracle_repr_type = oracle_solvert::repr_syntht::NEURAL_REPR;
  } else if (cmdline.get_value("oracle-repr") == "dt") {
    oracle_repr_type = oracle_solvert::repr_syntht::DT_REPR;
  } else if (cmdline.get_value("oracle-repr") == "symb_regr") {
    oracle_repr_type = oracle_solvert::repr_syntht::SYMB_REGRESSION_REPR;
  } else if (cmdline.get_value("oracle-repr") == "auto") {
      oracle_repr_type = oracle_solvert::repr_syntht::AUTO_REPR;
  }

  oracle_solvert::oracle_repr_optionst repr_options;
    repr_options.repr_type = oracle_repr_type;
  if (cmdline.isset("oracle-repr-freq")) {
      repr_options.frequency = std::stoi(cmdline.get_value("oracle-repr-freq"));
  } else {
      repr_options.frequency = 1;
  }
  if(cmdline.args.size() == 1) {
      repr_options.smtfilepath = cmdline.args.back();
  }

  repr_options.true_false_prediction = (oracle_repr_type == oracle_solvert::repr_syntht::DT_REPR);

// warning simplify won't work with oracles until freezing is implemented.
  if (cmdline.isset("negation-solver"))
  {
    if (cmdline.isset("bitblast") && cmdline.isset("simplify"))
    {
      satcheck_minisat_simplifiert satcheck_minisat_no_simplifier(message_handler);
      boolbvt subsolver(ns, satcheck_minisat_no_simplifier, message_handler);
      satcheck_minisat_simplifiert satcheck_minisat_no_simplifier2(message_handler);
      boolbvt subsolver2(ns, satcheck_minisat_no_simplifier2, message_handler);
      return negation_solver(in, subsolver, subsolver2, repr_options, message_handler);
    }
    else if (cmdline.isset("bitblast"))
    {
      satcheck_minisat_no_simplifiert satcheck_minisat_no_simplifier(message_handler);
      boolbvt subsolver(ns, satcheck_minisat_no_simplifier, message_handler);
      satcheck_minisat_no_simplifiert satcheck_minisat_no_simplifier2(message_handler);
      boolbvt subsolver2(ns, satcheck_minisat_no_simplifier2, message_handler);
      return negation_solver(in, subsolver, subsolver2, repr_options, message_handler);
    }
    else
    {
      smt2_dect subsolver(
          ns, "emu", "generated by emu",
          logic, smt2_dect::solvert::CVC4, message_handler);
      smt2_dect negation_subsolver(
          ns, "emu", "generated by emu for negation solver",
          logic, smt2_dect::solvert::CVC4, message_handler);
      return negation_solver(in, subsolver, negation_subsolver, repr_options, message_handler);
    }
  }
  else
  {
    if (cmdline.isset("bitblast") && cmdline.isset("simplify"))
    {
      satcheck_minisat_simplifiert satcheck_minisat_no_simplifier(message_handler);
      boolbvt subsolver(ns, satcheck_minisat_no_simplifier, message_handler);
      return solver(in, subsolver, repr_options, message_handler);
    }
    else if (cmdline.isset("bitblast"))
    {
      satcheck_minisat_no_simplifiert satcheck_minisat_no_simplifier(message_handler);
      boolbvt subsolver(ns, satcheck_minisat_no_simplifier, message_handler);
      return solver(in, subsolver, repr_options, message_handler);
    }
    else
    {
      smt2_dect subsolver(
          ns, "emu", "generated by emu",
          logic, smt2_dect::solvert::CVC4, message_handler);
      return solver(in, subsolver, repr_options, message_handler);
    }
  }
}

int smt2_frontend(const cmdlinet &cmdline)
{
  assert(cmdline.args.size()==1);
  std::ifstream in(cmdline.args.front());

  if(!in)
  {
    std::cerr << "Failed to open input file" << std::endl;
    return 10;
  }
  return smt2_frontend(cmdline, in);
}
