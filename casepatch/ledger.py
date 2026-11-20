"""Prototype of server-owned usage accounting. Not connected to payments or the free local app."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
from pathlib import Path
import re
import secrets
import sqlite3
import time

from .core import PolicyError
from .pricing import FREE_MONTHLY_CHARACTERS

# The proposed tariff is $10 / million characters = 10 microdollars / character.
MICRO_USD_PER_CHARACTER = 10
RESERVATION_SECONDS = 600


def opaque(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_-]{2,79}',value):
        raise PolicyError('OPAQUE_TECHNICAL_ID_REQUIRED')
    return value


def commitment(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{64}',value):
        raise PolicyError('KEYED_COMMITMENT_REQUIRED')
    return value


class UsageLedger:
    """Use one backend-owned file, not one ledger per caller or private runner."""
    def __init__(self,path):
        self.path=Path(path)
        if not self.path.exists():
            fd=__import__('os').open(self.path,__import__('os').O_CREAT|__import__('os').O_EXCL|__import__('os').O_WRONLY,0o600)
            __import__('os').close(fd)
        self.connection=sqlite3.connect(str(self.path),timeout=10,isolation_level=None)
        self.connection.row_factory=sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript('''
        CREATE TABLE IF NOT EXISTS accounts(account TEXT PRIMARY KEY, cap_micro_usd INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
          job TEXT PRIMARY KEY, account TEXT NOT NULL REFERENCES accounts(account),
          content TEXT NOT NULL, policy TEXT NOT NULL, model TEXT NOT NULL,
          period TEXT NOT NULL, characters INTEGER NOT NULL,
          state TEXT NOT NULL, expires REAL, receipt TEXT, output TEXT,
          UNIQUE(account,content,policy,model)
        );
        CREATE TABLE IF NOT EXISTS requests(
          account TEXT NOT NULL REFERENCES accounts(account), request TEXT NOT NULL,
          job TEXT NOT NULL REFERENCES jobs(job), PRIMARY KEY(account,request)
        );
        ''')

    def close(self):self.connection.close()

    @contextmanager
    def transaction(self):
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            self.connection.execute("UPDATE jobs SET state='released', expires=NULL WHERE state='reserved' AND (expires < ? OR period != ?)",(time.time(),dt.datetime.now(dt.timezone.utc).strftime('%Y-%m')))
            yield
            self.connection.execute('COMMIT')
        except Exception:
            self.connection.execute('ROLLBACK');raise

    def set_account(self,account,cap_micro_usd):
        opaque(account)
        if type(cap_micro_usd) is not int or not 0 <= cap_micro_usd <= 10**12:
            raise PolicyError('INVALID_SPEND_CAP')
        with self.transaction():
            period=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m')
            totals=self._totals(account,period)
            committed=self._charge(totals['committed_characters'])
            reserved_charge=self._charge(totals['committed_characters']+totals['reserved_characters'])
            if cap_micro_usd < max(committed,reserved_charge):raise PolicyError('CAP_BELOW_COMMITTED_OR_RESERVED_USAGE')
            self.connection.execute('INSERT INTO accounts VALUES(?,?) ON CONFLICT(account) DO UPDATE SET cap_micro_usd=excluded.cap_micro_usd',(account,cap_micro_usd))

    @staticmethod
    def _charge(characters):
        return max(0,characters-FREE_MONTHLY_CHARACTERS)*MICRO_USD_PER_CHARACTER

    def _totals(self,account,period):
        result={'committed_characters':0,'reserved_characters':0}
        for state,characters in self.connection.execute("SELECT state,SUM(characters) FROM jobs WHERE account=? AND period=? AND state IN ('reserved','committed') GROUP BY state",(account,period)):
            result[state+'_characters']=characters
        return result

    def _job(self,job):
        value=self.connection.execute('SELECT * FROM jobs WHERE job=?',(job,)).fetchone()
        if value is None:raise PolicyError('JOB_NOT_FOUND')
        return value

    @staticmethod
    def _view(row):
        return {'job_id':row['job'],'state':row['state'],'receipt_id':row['receipt'],
                'characters':row['characters'],'period':row['period'],'actual_charge_usd':'0.00',
                'pricing_status':'PROTOTYPE_NO_PAYMENTS'}

    def reserve(self,account,request,content,policy,model,characters):
        opaque(account);opaque(request);opaque(policy);opaque(model);commitment(content)
        if type(characters) is not int or not 0 < characters <= 10**9:raise PolicyError('INVALID_USAGE_COUNT')
        with self.transaction():
            customer=self.connection.execute('SELECT cap_micro_usd FROM accounts WHERE account=?',(account,)).fetchone()
            if customer is None:raise PolicyError('ACCOUNT_NOT_CONFIGURED')
            existing=self.connection.execute('SELECT job FROM requests WHERE account=? AND request=?',(account,request)).fetchone()
            row=self._job(existing['job']) if existing else self.connection.execute('SELECT * FROM jobs WHERE account=? AND content=? AND policy=? AND model=?',(account,content,policy,model)).fetchone()
            if row is not None:
                if (row['content'],row['policy'],row['model'],row['characters']) != (content,policy,model,characters):
                    raise PolicyError('IDEMPOTENCY_KEY_CONTENT_MISMATCH')
                if existing is None:self.connection.execute('INSERT INTO requests VALUES(?,?,?)',(account,request,row['job']))
                if row['state'] in {'committed','reserved'}:return self._view(row)
            period=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m')
            totals=self._totals(account,period)
            if self._charge(totals['committed_characters']+totals['reserved_characters']+characters) > customer['cap_micro_usd']:
                raise PolicyError('HARD_SPEND_CAP_REACHED')
            if row is None:
                job='job_'+secrets.token_hex(16)
                self.connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,'reserved',?,NULL,NULL)",(job,account,content,policy,model,period,characters,time.time()+RESERVATION_SECONDS))
                self.connection.execute('INSERT INTO requests VALUES(?,?,?)',(account,request,job))
            else:
                job=row['job']
                self.connection.execute("UPDATE jobs SET state='reserved',period=?,expires=? WHERE job=?",(period,time.time()+RESERVATION_SECONDS,job))
            return self._view(self._job(job))

    def commit(self,account,job,output,technical_checks_passed):
        opaque(account);opaque(job);commitment(output)
        if technical_checks_passed is not True:raise PolicyError('PASS_TECHNICAL_CHECKS_BEFORE_USAGE_COMMIT')
        with self.transaction():
            row=self._job(job)
            if row['account']!=account:raise PolicyError('JOB_NOT_FOUND')
            if row['state']=='committed':
                if row['output']!=output:raise PolicyError('RECEIPT_OUTPUT_MISMATCH')
                return self._view(row)
            if row['state']!='reserved':raise PolicyError('NO_ACTIVE_RESERVATION')
            period=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m')
            if row['period']!=period:
                raise PolicyError('RESERVE_AGAIN_IN_CURRENT_BILLING_PERIOD')
            receipt='receipt_'+secrets.token_hex(16)
            self.connection.execute("UPDATE jobs SET state='committed',expires=NULL,receipt=?,output=? WHERE job=?",(receipt,output,job))
            return self._view(self._job(job))

    def release(self,account,job):
        opaque(account);opaque(job)
        with self.transaction():
            row=self._job(job)
            if row['account']!=account:raise PolicyError('JOB_NOT_FOUND')
            if row['state']=='committed':raise PolicyError('COMMITTED_USAGE_NEEDS_CREDIT_WORKFLOW')
            self.connection.execute("UPDATE jobs SET state='released',expires=NULL WHERE job=?",(job,))
            return self._view(self._job(job))

    def usage(self,account):
        opaque(account)
        with self.transaction():
            customer=self.connection.execute('SELECT cap_micro_usd FROM accounts WHERE account=?',(account,)).fetchone()
            if customer is None:raise PolicyError('ACCOUNT_NOT_CONFIGURED')
            period=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m');totals=self._totals(account,period)
            return {'period':period,**totals,'proposed_charge_micro_usd':self._charge(totals['committed_characters']),
                    'cap_micro_usd':customer['cap_micro_usd'],'actual_charge_usd':'0.00','pricing_status':'PROTOTYPE_NO_PAYMENTS'}
